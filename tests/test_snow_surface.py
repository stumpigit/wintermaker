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


def test_multiscale_preserves_ridges_better_than_flat_blanket() -> None:
    width, height = 128, 128
    dem = np.linspace(2000, 2020, width, dtype=np.float32)[None, :] + np.linspace(
        0, 8, height, dtype=np.float32
    )[:, None]
    ridge = np.sin(np.linspace(0, 6 * np.pi, width, dtype=np.float32))[None, :] * 3.0
    dem = dem + ridge
    dem += np.random.default_rng(0).normal(0, 0.4, dem.shape).astype(np.float32)
    slope = np.zeros((height, width), dtype=np.float32)
    tpi = np.zeros((height, width), dtype=np.float32)
    aspect = np.zeros((height, width), dtype=np.float32)

    blanket = resolve_snow_surface_config(
        {
            "snow_surface": {
                "smoothing_sigma_m": 150,
                "peak_retention": 0.0,
                "surface_macro_smooth_sigma_m": 100,
                "valley_deposition_factor": 0.0,
                "ridge_scour_factor": 0.0,
            }
        }
    )
    multiscale = resolve_snow_surface_config(
        {
            "snow_surface": {
                "smoothing_sigma_m": 55,
                "micro_suppression": 0.82,
                "depression_fill": 0.92,
                "ridge_micro_retention": 0.40,
                "valley_deposition_factor": 0.0,
                "ridge_scour_factor": 0.0,
            }
        }
    )
    blanket_surface = compute_snow_surface_arrays(
        dem, slope, tpi, aspect, blanket, resolution_m=1.0
    )["snow_surface_dem"]
    multiscale_surface = compute_snow_surface_arrays(
        dem, slope, tpi, aspect, multiscale, resolution_m=1.0
    )["snow_surface_dem"]

    assert multiscale_surface.std() > blanket_surface.std()
    assert np.corrcoef(dem.ravel(), multiscale_surface.ravel())[0, 1] > np.corrcoef(
        dem.ravel(), blanket_surface.ravel()
    )[0, 1]


def test_snow_surface_never_below_dem() -> None:
    width, height = 96, 96
    dem = np.linspace(2000, 2040, width, dtype=np.float32)[None, :] + np.linspace(
        0, 12, height, dtype=np.float32
    )[:, None]
    ridge = np.sin(np.linspace(0, 10 * np.pi, width, dtype=np.float32))[None, :] * 8.0
    dem = dem + ridge
    slope = np.zeros((height, width), dtype=np.float32)
    tpi = np.zeros((height, width), dtype=np.float32)
    aspect = np.zeros((height, width), dtype=np.float32)

    cfg = resolve_snow_surface_config(
        {
            "snow_surface": {
                "base_snow_height_m": 4.0,
                "smoothing_sigma_m": 40,
                "micro_suppression": 0.82,
                "depression_fill": 0.92,
                "ridge_micro_retention": 0.40,
            }
        }
    )
    result = compute_snow_surface_arrays(
        dem, slope, tpi, aspect, cfg, resolution_m=0.5
    )

    assert (result["snow_thickness_m"] >= 0).all()
    assert (result["snow_surface_dem"] >= dem).all()


def test_peak_retention_zero_produces_smoother_surface_than_legacy() -> None:
    dem, slope, tpi, aspect = _flat_terrain(width=128, height=128)
    legacy = resolve_snow_surface_config({"snow_surface": {"peak_retention": 1.0}})
    smooth = resolve_snow_surface_config(
        {
            "snow_surface": {
                "peak_retention": 0.0,
                "surface_post_smooth_sigma_m": 10.0,
                "thickness_smoothing_sigma_m": 10.0,
            }
        }
    )
    legacy_result = compute_snow_surface_arrays(
        dem, slope, tpi, aspect, legacy, resolution_m=1.0
    )
    smooth_result = compute_snow_surface_arrays(
        dem, slope, tpi, aspect, smooth, resolution_m=1.0
    )
    legacy_std = float(legacy_result["snow_surface_dem"].std())
    smooth_std = float(smooth_result["snow_surface_dem"].std())
    assert smooth_std < legacy_std


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
