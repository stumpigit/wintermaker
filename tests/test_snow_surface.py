from __future__ import annotations

import numpy as np

from winter_ortho.rendering.compose import blend_hillshade_for_snow
from winter_ortho.snow_model.surface import compute_snow_surface_arrays, resolve_snow_surface_config


def _flat_terrain(
    height: int = 64,
    width: int = 64,
    *,
    base_elev: float = 2000.0,
    depression_depth: float = 1.5,
    ridge_height: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dem = np.full((height, width), base_elev, dtype=np.float32)
    dem[20:30, 20:30] -= depression_depth
    dem[40:45, 40:50] += ridge_height
    slope = np.zeros((height, width), dtype=np.float32)
    tpi = np.zeros((height, width), dtype=np.float32)
    tpi[20:30, 20:30] = -2.0
    tpi[40:45, 40:50] = 2.0
    aspect = np.zeros((height, width), dtype=np.float32)
    return dem, slope, tpi, aspect


def test_resolve_snow_surface_config_override() -> None:
    cfg = resolve_snow_surface_config({"snow_surface": {"base_snow_height_m": 1.5}}, snow_height_m=3.0)
    assert cfg["base_snow_height_m"] == 3.0
    assert cfg["max_accumulation_slope_deg"] == 30.0


def test_depression_gets_extra_thickness_on_flat_terrain() -> None:
    dem, slope, tpi, aspect = _flat_terrain()
    cfg = resolve_snow_surface_config({"snow_surface": {"base_snow_height_m": 2.0}})
    result = compute_snow_surface_arrays(
        dem,
        slope,
        tpi,
        aspect,
        cfg,
        resolution_m=1.0,
    )

    depression = result["snow_thickness_m"][25, 25]
    flat = result["snow_thickness_m"][10, 10]
    assert depression > flat
    assert depression >= 2.0


def test_ridge_gets_reduced_thickness() -> None:
    dem, slope, tpi, aspect = _flat_terrain()
    cfg = resolve_snow_surface_config({"snow_surface": {"base_snow_height_m": 2.0}})
    result = compute_snow_surface_arrays(
        dem,
        slope,
        tpi,
        aspect,
        cfg,
        resolution_m=1.0,
    )

    ridge = result["snow_thickness_m"][42, 45]
    flat = result["snow_thickness_m"][10, 10]
    assert ridge < flat


def test_steep_slope_skips_accumulation_smoothing() -> None:
    dem = np.full((32, 32), 2000.0, dtype=np.float32)
    slope = np.zeros((32, 32), dtype=np.float32)
    slope[10:20, 10:20] = 45.0
    tpi = np.zeros((32, 32), dtype=np.float32)
    aspect = np.zeros((32, 32), dtype=np.float32)

    cfg = resolve_snow_surface_config({"snow_surface": {"base_snow_height_m": 2.0}})
    result = compute_snow_surface_arrays(
        dem,
        slope,
        tpi,
        aspect,
        cfg,
        resolution_m=1.0,
    )

    assert result["accumulation_mask"][5, 5] == 1
    assert result["accumulation_mask"][15, 15] == 0
    assert result["snow_thickness_m"][15, 15] < result["snow_thickness_m"][5, 5]


def test_blend_hillshade_prefers_snow_surface_on_flat_snowy_pixels() -> None:
    base = np.full((8, 8), 0.2, dtype=np.float32)
    snow = np.full((8, 8), 0.8, dtype=np.float32)
    fraction = np.full((8, 8), 1.0, dtype=np.float32)
    slope = np.zeros((8, 8), dtype=np.float32)

    blended = blend_hillshade_for_snow(
        base,
        snow,
        fraction,
        slope,
        max_accumulation_slope_deg=30.0,
    )
    assert np.allclose(blended, 0.8)

    slope_steep = np.full((8, 8), 50.0, dtype=np.float32)
    blended_steep = blend_hillshade_for_snow(
        base,
        snow,
        fraction,
        slope_steep,
        max_accumulation_slope_deg=30.0,
    )
    assert np.allclose(blended_steep, 0.2)
