from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage
from skimage.morphology import binary_closing, disk

from winter_ortho.preprocessing.rasterize_vectors import apply_priority
from winter_ortho.rendering.base import to_float_rgb
from winter_ortho.rendering.relief import luminance
from winter_ortho.utils.progress import PipelineProgress


def compute_summer_appearance(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Heuristic forest / meadow / rock scores from summer orthophoto (0–1)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = luminance(rgb)
    green_excess = g - np.maximum(r, b)
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = (max_c - min_c) / np.maximum(max_c, 1e-4)

    forest = (
        np.clip((green_excess - 0.015) / 0.10, 0.0, 1.0)
        * np.clip(sat / 0.22, 0.25, 1.0)
        * (1.0 - np.clip((lum - 0.42) / 0.35, 0.0, 0.85))
    )
    meadow = (
        np.clip(green_excess / 0.12, 0.0, 1.0)
        * np.clip(lum / 0.38, 0.35, 1.0)
        * np.clip(sat / 0.28, 0.2, 1.0)
    )
    rock = (
        np.clip(1.0 - green_excess / 0.09, 0.0, 1.0)
        * np.clip(1.0 - np.clip(g - 0.22, 0.0, 1.0) / 0.38, 0.25, 1.0)
        * (0.50 + 0.50 * np.clip(sat / 0.14, 0.0, 1.0))
    )
    return {
        "forest": forest.astype(np.float32),
        "meadow": meadow.astype(np.float32),
        "rock": rock.astype(np.float32),
        "green_excess": green_excess.astype(np.float32),
    }


def _protected_mask(class_masks: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    shape = next(iter(class_masks.values())).shape
    out = np.zeros(shape, dtype=bool)
    for name in names:
        mask = class_masks.get(name)
        if mask is not None:
            out |= mask > 0
    return out


def _recompute_open_land(class_masks: dict[str, np.ndarray], priority: list[str]) -> None:
    occupied = np.zeros_like(next(iter(class_masks.values())), dtype=bool)
    for name in priority:
        if name in ("open_land_mask", "special_area_mask"):
            continue
        occupied |= class_masks.get(name, np.zeros_like(occupied, dtype=np.uint8)) > 0
    class_masks["open_land_mask"] = (~occupied).astype(np.uint8)


def reconcile_masks_with_summer(
    summer_rgb: np.ndarray,
    class_masks: dict[str, np.ndarray],
    class_rules: dict[str, Any],
    *,
    progress: PipelineProgress | None = None,
) -> dict[str, np.ndarray]:
    """
    Align TLM masks with visible summer orthophoto.

    - Rock: drop TLM/DEM rock where summer shows vegetation or no bare rock.
    - Forest: fill interior gaps and spectral forest patches inside the canopy envelope.
    """
    cfg = class_rules.get("summer_reconcile", {})
    if not cfg.get("enabled", True):
        return class_masks

    rgb = summer_rgb
    if rgb.max() > 1.0:
        rgb = to_float_rgb(rgb)
    scores = compute_summer_appearance(rgb)

    priority = class_rules["priority"]
    mask_values = class_rules["mask_values"]
    protected = _protected_mask(
        class_masks,
        cfg.get(
            "protect_masks",
            ["water_mask", "building_mask", "road_mask", "path_mask"],
        ),
    )

    rock_cfg = cfg.get("rock", {})
    forest_cfg = cfg.get("forest", {})
    rock_mask = class_masks.get("rock_or_bare_ground_mask", np.zeros_like(protected, dtype=np.uint8))
    forest_mask = class_masks.get("forest_mask", np.zeros_like(protected, dtype=np.uint8))
    forest_before = int((forest_mask > 0).sum())

    min_rock = float(rock_cfg.get("min_rock_score", 0.38))
    max_green = float(rock_cfg.get("max_green_excess", 0.055))
    min_meadow = float(rock_cfg.get("min_meadow_score", 0.42))
    min_forest_on_rock = float(rock_cfg.get("min_forest_score", 0.32))

    rock_pixels = rock_mask > 0
    summer_not_rock = (
        (scores["meadow"] >= min_meadow)
        | (scores["forest"] >= min_forest_on_rock)
        | (scores["green_excess"] > max_green)
        | (scores["rock"] < min_rock)
    )
    remove_rock = rock_pixels & summer_not_rock
    forest_bool = forest_mask > 0
    near_forest = (
        ndimage.binary_dilation(forest_bool, iterations=int(rock_cfg.get("reassign_forest_px", 5)))
        if forest_bool.any()
        else forest_bool
    )
    reassign_forest = (
        remove_rock
        & (scores["forest"] >= min_forest_on_rock)
        & near_forest
    )

    rock_removed = int(remove_rock.sum())
    rock_mask = rock_mask.copy()
    rock_mask[remove_rock] = 0
    class_masks["rock_or_bare_ground_mask"] = rock_mask

    forest_mask = forest_mask.copy()
    forest_mask[reassign_forest & ~protected] = 1

    if forest_cfg.get("fill_holes", True):
        radius = int(forest_cfg.get("hole_close_radius_px", 6))
        min_forest = float(forest_cfg.get("min_forest_score", 0.34))
        loose_forest = float(forest_cfg.get("hole_min_forest_score", 0.22))
        structure = disk(max(1, radius)) if radius > 0 else None
        forest_bool = forest_mask > 0
        if structure is not None:
            envelope = binary_closing(forest_bool, structure)
        else:
            envelope = forest_bool
        filled = ndimage.binary_fill_holes(envelope)
        holes = filled & ~forest_bool & ~protected
        add_forest = holes & (scores["forest"] >= loose_forest)
        forest_mask[add_forest] = 1

        expand_px = int(forest_cfg.get("interior_expand_px", 3))
        if expand_px > 0:
            interior = ndimage.binary_dilation(filled, iterations=expand_px)
            fringe = interior & ~(forest_mask > 0) & ~protected
            forest_mask[fringe & (scores["forest"] >= min_forest)] = 1

    forest_added = int((forest_mask > 0).sum() - forest_before)

    class_masks["forest_mask"] = forest_mask
    _recompute_open_land(class_masks, priority)

    if progress:
        progress.substep(
            f"Summer reconcile: {rock_removed} rock px removed, "
            f"{forest_added} forest px added (orthophoto authority)"
        )

    return class_masks


def write_reconciled_masks(
    class_masks: dict[str, np.ndarray],
    class_rules: dict[str, Any],
    *,
    tlm_masks_path,
    intermediate_dir,
    transform,
    crs: str,
) -> None:
    from winter_ortho.utils.raster import write_cog

    priority = class_rules["priority"]
    mask_values = class_rules["mask_values"]
    labeled = apply_priority(class_masks, priority, mask_values)
    write_cog(str(tlm_masks_path), labeled.astype(np.uint8), transform=transform, crs=crs, nodata=0)
    for name, mask in class_masks.items():
        write_cog(
            str(intermediate_dir / f"{name}.tif"),
            mask.astype(np.uint8),
            transform=transform,
            crs=crs,
            nodata=0,
        )
