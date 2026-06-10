from __future__ import annotations

from typing import Any

from winter_ortho.features.terrain import SNOW_TERRAIN_BANDS, compute_terrain_features, load_terrain_bands
from winter_ortho.masks.build_masks import build_tile_masks
from winter_ortho.masks.protect_masks import build_protect_mask
from winter_ortho.masks.summer_reconcile import reconcile_masks_with_summer, write_reconciled_masks
from winter_ortho.rendering.base import to_float_rgb
from winter_ortho.preprocessing.align import harmonize_tile
from winter_ortho.preprocessing.tiling import get_tile_grid
from winter_ortho.qa.geometry_checks import run_qa
from winter_ortho.rendering.compose import render_winter_tile
from winter_ortho.snow_model.rules import compute_snow_layers
from winter_ortho.snow_model.surface import compute_snow_surface
from winter_ortho.utils.config import load_class_rules, load_config, load_profile
from winter_ortho.utils.paths import tile_paths
from winter_ortho.utils.progress import PipelineProgress
from winter_ortho.utils.raster import read_raster

PIPELINE_STEPS = [
    ("harmonize", "Datenharmonisierung", "Orthofoto + DEM auf gemeinsames Raster"),
    ("masks", "Geometrische Masken", "TLM3D-Vektoren rasterisieren"),
    ("terrain", "Terrain-Features", "Hang, Exposition, Hillshade aus DEM"),
    ("snow_surface", "Schneeoberfläche", "DEM-Glättung und Schneedicke in Metern"),
    ("snow", "Schneebedeckungsmodell", "snow_fraction und Zwischenlayer"),
    ("render", "Winter-Rendering", "Regelbasiertes Winter-Orthofoto"),
    ("qa", "Qualitätskontrolle", "Geometrie- und Plausibilitätschecks"),
]


def _load_class_masks(paths) -> dict[str, Any]:
    names = [
        "building_mask",
        "road_mask",
        "path_mask",
        "water_mask",
        "forest_mask",
        "settlement_mask",
        "open_land_mask",
        "rock_or_bare_ground_mask",
        "special_area_mask",
    ]
    masks = {}
    for name in names:
        mask_path = paths.intermediate_dir / f"{name}.tif"
        data, _ = read_raster(str(mask_path))
        masks[name] = data[0] if data.ndim == 3 else data
    return masks


def _load_terrain(paths) -> dict[str, Any]:
    return load_terrain_bands(paths, SNOW_TERRAIN_BANDS)


def _load_snow_surface(paths) -> dict[str, Any] | None:
    if not paths.snow_thickness_m.exists():
        return None
    layers: dict[str, np.ndarray] = {}
    for name in ("snow_surface_dem", "snow_thickness_m", "accumulation_mask"):
        data, _ = read_raster(str(getattr(paths, name)))
        layers[name] = data[0] if data.ndim == 3 else data
    return layers


def _load_snow_layers(paths) -> dict[str, Any]:
    names = [
        "snow_fraction",
        "snow_brightness",
        "snow_texture_strength",
        "rock_visibility",
        "forest_snow_intensity",
        "road_visibility",
        "roof_snow_intensity",
        "ice_probability",
    ]
    layers = {}
    for name in names:
        data, _ = read_raster(str(getattr(paths, name)))
        layers[name] = data[0] if data.ndim == 3 else data
    return layers


def run_harmonize(
    tile_id: str,
    config_path: str | None = None,
    *,
    progress: PipelineProgress | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    paths = tile_paths(config, tile_id)
    return harmonize_tile(config, paths, progress=progress)


def run_masks(
    tile_id: str,
    config_path: str | None = None,
    *,
    progress: PipelineProgress | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    class_rules = load_class_rules()
    paths = tile_paths(config, tile_id)
    grid = get_tile_grid(config, tile_id)
    class_masks = build_tile_masks(config, class_rules, paths, progress=progress)
    if paths.rgb_summer.exists():
        if progress:
            progress.substep("Reconciling TLM masks with summer orthophoto")
        summer_raw, _ = read_raster(str(paths.rgb_summer))
        summer_rgb = to_float_rgb(summer_raw)
        class_masks = reconcile_masks_with_summer(
            summer_rgb, class_masks, class_rules, progress=progress
        )
        write_reconciled_masks(
            class_masks,
            class_rules,
            tlm_masks_path=paths.tlm_masks,
            intermediate_dir=paths.intermediate_dir,
            transform=grid.transform,
            crs=grid.crs,
        )
    if progress:
        progress.substep("Building geometry protection mask")
    build_protect_mask(
        class_rules,
        paths,
        class_masks,
        transform=grid.transform,
        crs=grid.crs,
    )
    return {"mask_count": len(class_masks)}


def run_terrain(
    tile_id: str,
    config_path: str | None = None,
    *,
    progress: PipelineProgress | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    paths = tile_paths(config, tile_id)
    features = compute_terrain_features(config, paths, progress=progress)
    return {"feature_count": len(features)}


def run_snow_surface(
    tile_id: str,
    profile_name: str = "davos",
    config_path: str | None = None,
    *,
    snow_height_m: float | None = None,
    progress: PipelineProgress | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    profile = load_profile(profile_name)
    paths = tile_paths(config, tile_id)
    if progress:
        progress.substep("Loading terrain features")
    terrain = _load_terrain(paths)
    arrays = compute_snow_surface(
        config,
        profile,
        paths,
        terrain,
        snow_height_m=snow_height_m,
        progress=progress,
    )
    return {"layer_count": len(arrays)}


def run_snow(
    tile_id: str,
    profile_name: str = "davos",
    config_path: str | None = None,
    *,
    snow_height_m: float | None = None,
    progress: PipelineProgress | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    profile = load_profile(profile_name)
    paths = tile_paths(config, tile_id)
    if progress:
        progress.substep("Loading masks, terrain, and snow surface")
    class_masks = _load_class_masks(paths)
    terrain = _load_terrain(paths)
    snow_surface = _load_snow_surface(paths)
    if snow_surface is None:
        if progress:
            progress.substep("Snow surface missing — computing inline")
        snow_surface = compute_snow_surface(
            config,
            profile,
            paths,
            terrain,
            snow_height_m=snow_height_m,
            progress=progress,
        )
    layers = compute_snow_layers(
        config,
        profile,
        paths,
        class_masks,
        terrain,
        snow_thickness=snow_surface.get("snow_thickness_m"),
        progress=progress,
    )
    return {"layer_count": len(layers)}


def run_render(
    tile_id: str,
    profile_name: str = "davos",
    config_path: str | None = None,
    *,
    progress: PipelineProgress | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    profile = load_profile(profile_name)
    paths = tile_paths(config, tile_id)
    if progress:
        progress.substep("Loading masks, snow layers, terrain, snow surface")
    class_masks = _load_class_masks(paths)
    terrain = _load_terrain(paths)
    snow_layers = _load_snow_layers(paths)
    snow_surface = _load_snow_surface(paths)
    render_winter_tile(
        config,
        profile,
        paths,
        class_masks,
        snow_layers,
        terrain,
        snow_surface=snow_surface,
        progress=progress,
    )
    return {"output": str(paths.winter_rgb)}


def run_qa_step(
    tile_id: str,
    config_path: str | None = None,
    *,
    progress: PipelineProgress | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    paths = tile_paths(config, tile_id)
    grid = get_tile_grid(config, tile_id)
    class_masks = _load_class_masks(paths)
    return run_qa(
        config,
        paths,
        class_masks,
        transform=grid.transform,
        crs=grid.crs,
        progress=progress,
    )


def run_all(
    tile_id: str,
    profile_name: str = "davos",
    config_path: str | None = None,
    *,
    progress: PipelineProgress | None = None,
) -> dict[str, Any]:
    total = len(PIPELINE_STEPS)
    results: dict[str, Any] = {"tile_id": tile_id, "profile": profile_name}

    step_fns = [
        lambda: run_harmonize(tile_id, config_path, progress=progress),
        lambda: run_masks(tile_id, config_path, progress=progress),
        lambda: run_terrain(tile_id, config_path, progress=progress),
        lambda: run_snow_surface(tile_id, profile_name, config_path, progress=progress),
        lambda: run_snow(tile_id, profile_name, config_path, progress=progress),
        lambda: run_render(tile_id, profile_name, config_path, progress=progress),
        lambda: run_qa_step(tile_id, config_path, progress=progress),
    ]

    for idx, ((key, title, detail), fn) in enumerate(zip(PIPELINE_STEPS, step_fns), start=1):
        if progress:
            progress.advance(title)
            progress.step_begin(idx, total, title, detail)
        result = fn()
        results[key] = result
        if progress:
            summary = _step_summary(key, result)
            progress.step_end(title, summary)

    return results


def _step_summary(step_key: str, result: dict[str, Any]) -> str:
    if step_key == "harmonize":
        return (
            f"{result['width']}×{result['height']} px, "
            f"nodata {result.get('nodata_fraction', 0):.1%}"
        )
    if step_key == "masks":
        return f"{result['mask_count']} masks"
    if step_key == "terrain":
        return f"{result['feature_count']} features"
    if step_key == "snow_surface":
        return f"{result['layer_count']} layers"
    if step_key == "snow":
        return f"{result['layer_count']} layers"
    if step_key == "render":
        return str(result.get("output", ""))
    if step_key == "qa":
        status = "PASS" if result.get("overall_pass") else "FAIL"
        return status
    return ""
