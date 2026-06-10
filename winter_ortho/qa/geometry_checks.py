from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np

from winter_ortho.utils.paths import TilePaths
from winter_ortho.utils.progress import PipelineProgress
from winter_ortho.utils.raster import read_raster, write_cog

QA_CHECKS = [
    "building_edges",
    "road_visibility",
    "water_boundary",
    "forest_boundary",
    "no_hallucination",
]


def run_qa(
    config: dict[str, Any],
    paths: TilePaths,
    class_masks: dict[str, np.ndarray],
    *,
    transform,
    crs: str,
    progress: PipelineProgress | None = None,
) -> dict[str, Any]:
    qa_cfg = config.get("qa", {})
    if progress:
        progress.substep("Running QA checks: " + ", ".join(QA_CHECKS))

    summer, _ = read_raster(str(paths.rgb_summer))
    winter, _ = read_raster(str(paths.winter_rgb))
    summer_rgb = _to_hwc(summer)
    winter_rgb = _to_hwc(winter)

    building_mask = class_masks["building_mask"] > 0
    road_mask = class_masks["road_mask"] > 0
    water_mask = class_masks["water_mask"] > 0
    forest_mask = class_masks["forest_mask"] > 0

    building_score = _edge_stability(
        summer_rgb,
        winter_rgb,
        building_mask,
        tolerance=float(qa_cfg.get("building_edge_tolerance", 0.35)),
    )
    road_score = _road_visibility(
        winter_rgb,
        road_mask,
        min_brightness=float(qa_cfg.get("road_brightness_min", 0.25)),
    )
    water_score = _water_boundary_stability(
        summer_rgb,
        winter_rgb,
        water_mask,
        tolerance=float(qa_cfg.get("water_boundary_tolerance", 0.15)),
    )
    forest_score = _boundary_stability(
        summer_rgb,
        winter_rgb,
        forest_mask,
        tolerance=float(qa_cfg.get("forest_boundary_tolerance", 0.30)),
    )
    hallucination_score = _hallucination_check(
        summer_rgb,
        winter_rgb,
        protected_mask=building_mask | water_mask,
        tolerance=float(qa_cfg.get("hallucination_tolerance", 0.40)),
    )

    quality_flags = np.zeros(summer_rgb.shape[:2], dtype=np.uint8)
    if building_score["pass"]:
        quality_flags[building_mask] |= 1
    if road_score["pass"]:
        quality_flags[road_mask] |= 2
    if water_score["pass"]:
        quality_flags[water_mask] |= 4
    if forest_score["pass"]:
        quality_flags[forest_mask] |= 8

    write_cog(
        str(paths.quality_flags),
        quality_flags,
        transform=transform,
        crs=crs,
        nodata=0,
    )

    checks = {
        "building_edges": building_score,
        "road_visibility": road_score,
        "water_boundary": water_score,
        "forest_boundary": forest_score,
        "no_hallucination": hallucination_score,
    }
    overall_pass = all(c["pass"] for c in checks.values())

    if progress:
        for name, result in checks.items():
            icon = "[green]✓[/green]" if result["pass"] else "[red]✗[/red]"
            score = result.get("score", "—")
            progress.substep(f"{icon} {name}: score={score}")
        progress.substep(f"Writing {paths.quality_flags.name}, qa_report.json")

    report = {
        "tile_id": paths.tile_id,
        "checks": checks,
        "overall_pass": overall_pass,
    }
    paths.qa_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _to_hwc(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[0] <= 4:
        return np.moveaxis(array[:3], 0, -1)
    return array[..., :3]


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _edge_stability(
    summer: np.ndarray,
    winter: np.ndarray,
    mask: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, Any]:
    if not mask.any():
        return {"pass": True, "score": 1.0, "message": "no buildings in tile"}
    summer_edges = cv2.Canny(_gray(summer), 50, 150)
    winter_edges = cv2.Canny(_gray(winter), 50, 150)
    region = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    overlap = (summer_edges & winter_edges & region).sum()
    union = ((summer_edges | winter_edges) & region).sum() + 1e-6
    score = float(overlap / union)
    return {"pass": score >= (1.0 - tolerance), "score": score}


def _road_visibility(
    winter: np.ndarray,
    mask: np.ndarray,
    *,
    min_brightness: float,
) -> dict[str, Any]:
    if not mask.any():
        return {"pass": True, "score": 1.0, "message": "no roads in tile"}
    gray = _gray(winter) / 255.0
    score = float(gray[mask].mean())
    return {"pass": score >= min_brightness, "score": score}


def _water_boundary_stability(
    summer: np.ndarray,
    winter: np.ndarray,
    mask: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, Any]:
    if not mask.any():
        return {"pass": True, "score": 1.0, "message": "no water in tile"}
    boundary = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) - mask.astype(
        np.uint8
    )
    diff = np.abs(_gray(summer).astype(np.float32) - _gray(winter).astype(np.float32)) / 255.0
    score = float(1.0 - diff[boundary > 0].mean())
    return {"pass": score >= (1.0 - tolerance), "score": score}


def _boundary_stability(
    summer: np.ndarray,
    winter: np.ndarray,
    mask: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, Any]:
    if not mask.any():
        return {"pass": True, "score": 1.0, "message": "no forest in tile"}
    boundary = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) - mask.astype(
        np.uint8
    )
    diff = np.abs(_gray(summer).astype(np.float32) - _gray(winter).astype(np.float32)) / 255.0
    score = float(1.0 - diff[boundary > 0].mean())
    return {"pass": score >= (1.0 - tolerance), "score": score}


def _hallucination_check(
    summer: np.ndarray,
    winter: np.ndarray,
    *,
    protected_mask: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    diff = np.abs(summer.astype(np.float32) - winter.astype(np.float32)).mean(axis=-1) / 255.0
    unprotected = ~protected_mask
    if not unprotected.any():
        return {"pass": True, "score": 1.0}
    mean_change = float(diff[unprotected].mean())
    return {"pass": mean_change <= tolerance, "score": 1.0 - mean_change}
