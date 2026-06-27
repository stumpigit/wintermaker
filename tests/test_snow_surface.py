from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

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


def test_slope_boundary_transitions_smoothly_without_steps() -> None:
    width = 128
    dem = np.full((width, width), 2000.0, dtype=np.float32)
    slope = np.zeros((width, width), dtype=np.float32)
    slope[:, width // 2 :] = 50.0
    tpi = np.zeros((width, width), dtype=np.float32)
    aspect = np.zeros((width, width), dtype=np.float32)

    cfg = resolve_snow_surface_config(
        {
            "snow_surface": {
                "base_snow_height_m": 4.0,
                "max_accumulation_slope_deg": 35.0,
                "accumulation_transition_deg": 10.0,
                "accumulation_blend_sigma_m": 15.0,
                "smoothing_sigma_m": 30.0,
                "micro_suppression": 0.82,
                "surface_post_smooth_sigma_m": 10.0,
            }
        }
    )
    result = compute_snow_surface_arrays(
        dem, slope, tpi, aspect, cfg, resolution_m=1.0
    )
    snow = result["snow_surface_dem"]
    thickness = result["snow_thickness_m"]

    flat = snow[:, 10]
    steep = snow[:, width // 2 + 10]

    assert flat.mean() > steep.mean() + 1.0
    assert (thickness >= 0).all()
    assert (snow >= dem).all()
    assert result["snow_thickness_m"][:, width // 2 + 10].mean() < 0.05
    assert result["snow_thickness_m"][:, 10].mean() > 3.0


def test_flat_deck_stays_homogeneous_before_steep_boundary() -> None:
    """Gentle accumulation keeps full deck until slope increases; steep has no overlap."""
    width = 96
    dem = np.full((width, width), 2000.0, dtype=np.float32)
    slope = np.full((width, width), 15.0, dtype=np.float32)
    slope[:, width // 2 :] = 55.0
    tpi = np.zeros((width, width), dtype=np.float32)
    aspect = np.zeros((width, width), dtype=np.float32)

    cfg = resolve_snow_surface_config(
        {
            "snow_surface": {
                "base_snow_height_m": 2.0,
                "max_accumulation_slope_deg": 35.0,
                "accumulation_transition_deg": 20.0,
                "accumulation_edge_feather_m": 40.0,
            }
        }
    )
    result = compute_snow_surface_arrays(
        dem, slope, tpi, aspect, cfg, resolution_m=1.0
    )
    thick = result["snow_thickness_m"]
    snow = result["snow_surface_dem"]

    interior = thick[:, 10].mean()
    beyond_feather = thick[:, width // 2 - 45].mean()
    near_cliff = thick[:, width // 2 - 5].mean()
    steep = thick[:, width // 2 + 5].mean()
    assert interior >= 1.9
    assert np.isclose(interior, beyond_feather, atol=0.05)
    assert near_cliff < interior * 0.35
    assert steep < 0.1
    assert (snow >= dem).all()


def test_edge_feather_preserves_interior_blanket() -> None:
    width = 96
    dem = np.full((width, width), 2000.0, dtype=np.float32)
    dem += np.sin(np.linspace(0, 10 * np.pi, width, dtype=np.float32))[None, :] * 6.0
    slope = np.full((width, width), 20.0, dtype=np.float32)
    slope[:, width // 2 :] = 48.0
    tpi = np.zeros((width, width), dtype=np.float32)
    aspect = np.zeros((width, width), dtype=np.float32)

    base_cfg = {
        "base_snow_height_m": 7.0,
        "max_accumulation_slope_deg": 35.0,
        "accumulation_transition_deg": 12.0,
        "accumulation_blend_sigma_m": 10.0,
        "smoothing_sigma_m": 30.0,
        "micro_suppression": 0.82,
    }
    with_feather = resolve_snow_surface_config(
        {"snow_surface": {**base_cfg, "accumulation_edge_feather_m": 30.0}}
    )
    without_feather = resolve_snow_surface_config(
        {"snow_surface": {**base_cfg, "accumulation_edge_feather_m": 0.0}}
    )
    thick_feather = compute_snow_surface_arrays(
        dem, slope, tpi, aspect, with_feather, resolution_m=1.0
    )["snow_thickness_m"]
    thick_plain = compute_snow_surface_arrays(
        dem, slope, tpi, aspect, without_feather, resolution_m=1.0
    )["snow_thickness_m"]

    interior_col = 10
    assert thick_feather[:, interior_col].mean() >= 6.5
    assert np.isclose(thick_feather[:, interior_col].mean(), thick_plain[:, interior_col].mean())


def test_shaped_layer_below_thickness_does_not_create_depressions() -> None:
    width, height = 64, 64
    dem = np.full((height, width), 2000.0, dtype=np.float32)
    dem += np.sin(np.linspace(0, 12 * np.pi, width, dtype=np.float32))[None, :] * 10.0
    slope = np.full((height, width), 12.0, dtype=np.float32)
    tpi = np.zeros((height, width), dtype=np.float32)
    aspect = np.zeros((height, width), dtype=np.float32)

    cfg = resolve_snow_surface_config(
        {
            "snow_surface": {
                "base_snow_height_m": 2.0,
                "max_accumulation_slope_deg": 35.0,
                "accumulation_edge_feather_m": 30.0,
                "accumulation_blend_sigma_m": 15.0,
            }
        }
    )
    result = compute_snow_surface_arrays(
        dem, slope, tpi, aspect, cfg, resolution_m=0.5
    )
    thick = result["snow_thickness_m"]
    assert (thick < 1.0).sum() == 0
    assert (thick >= 1.9).mean() > 0.9


def test_micro_suppression_does_not_create_holes_on_flat_accumulation() -> None:
    width, height = 96, 96
    dem = np.full((height, width), 2000.0, dtype=np.float32)
    dem += np.sin(np.linspace(0, 14 * np.pi, width, dtype=np.float32))[None, :] * 6.0
    slope = np.full((height, width), 15.0, dtype=np.float32)
    tpi = np.zeros((height, width), dtype=np.float32)
    aspect = np.zeros((height, width), dtype=np.float32)

    cfg = resolve_snow_surface_config(
        {
            "snow_surface": {
                "base_snow_height_m": 2.0,
                "max_accumulation_slope_deg": 35.0,
                "accumulation_transition_deg": 12.0,
                "accumulation_blend_sigma_m": 10.0,
            }
        }
    )
    result = compute_snow_surface_arrays(
        dem, slope, tpi, aspect, cfg, resolution_m=0.5
    )
    thick = result["snow_thickness_m"]
    assert (thick >= 1.9).mean() > 0.95
    assert (thick < 0.2).sum() == 0


def test_noisy_pixel_slope_does_not_punch_holes_when_blend_smoothing_enabled() -> None:
    width, height = 128, 128
    dem = np.full((height, width), 2000.0, dtype=np.float32)
    dem += np.sin(np.linspace(0, 16 * np.pi, width, dtype=np.float32))[None, :] * 4.0
    dem += np.random.default_rng(0).normal(0, 0.8, dem.shape).astype(np.float32)
    dz_dy, dz_dx = np.gradient(dem, 2.0, 2.0)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))).astype(np.float32)
    tpi = np.zeros((height, width), dtype=np.float32)
    aspect = np.zeros((height, width), dtype=np.float32)

    cfg = resolve_snow_surface_config(
        {
            "snow_surface": {
                "base_snow_height_m": 2.0,
                "max_accumulation_slope_deg": 35.0,
                "accumulation_transition_deg": 20.0,
                "accumulation_edge_feather_m": 45.0,
                "accumulation_blend_sigma_m": 15.0,
                "micro_suppression": 0.90,
                "smoothing_sigma_m": 100.0,
            }
        }
    )
    result = compute_snow_surface_arrays(
        dem, slope, tpi, aspect, cfg, resolution_m=2.0
    )
    thick = result["snow_thickness_m"]
    gentle = slope < 30.0

    assert (thick[gentle] < 1.0).sum() == 0
    assert thick[gentle].mean() > 1.9


def test_snow_amount_scales_thickness_without_changing_reference_height() -> None:
    dem, slope, tpi, aspect = _flat_terrain()
    full = compute_snow_surface_arrays(
        dem,
        slope,
        tpi,
        aspect,
        resolve_snow_surface_config(
            {"snow_surface": {"base_snow_height_m": 2.0, "snow_amount": 1.0}}
        ),
        resolution_m=1.0,
    )["snow_thickness_m"]
    reduced = compute_snow_surface_arrays(
        dem,
        slope,
        tpi,
        aspect,
        resolve_snow_surface_config(
            {"snow_surface": {"base_snow_height_m": 2.0, "snow_amount": 0.4}}
        ),
        resolution_m=1.0,
    )["snow_thickness_m"]

    flat_full = float(full[10, 10])
    flat_reduced = float(reduced[10, 10])
    assert np.isclose(flat_full, 2.0, atol=0.05)
    assert np.isclose(flat_reduced, 0.8, atol=0.05)
    assert np.isclose(flat_reduced / flat_full, 0.4, atol=0.02)


def test_base_snow_height_shifts_surface_on_noisy_terrain() -> None:
    width, height = 96, 96
    dem = np.full((height, width), 2000.0, dtype=np.float32)
    dem += np.sin(np.linspace(0, 12 * np.pi, width, dtype=np.float32))[None, :] * 5.0
    dem += np.random.default_rng(1).normal(0, 0.6, dem.shape).astype(np.float32)
    slope = np.full((height, width), 12.0, dtype=np.float32)
    tpi = np.zeros((height, width), dtype=np.float32)
    aspect = np.zeros((height, width), dtype=np.float32)

    base_cfg = {
        "max_accumulation_slope_deg": 35.0,
        "accumulation_transition_deg": 20.0,
        "accumulation_edge_feather_m": 45.0,
        "accumulation_blend_sigma_m": 15.0,
        "micro_suppression": 0.90,
        "smoothing_sigma_m": 100.0,
        "leveling_full_slope_deg": 30.0,
        "leveling_end_slope_deg": 35.0,
    }
    low = compute_snow_surface_arrays(
        dem,
        slope,
        tpi,
        aspect,
        resolve_snow_surface_config({"snow_surface": {**base_cfg, "base_snow_height_m": 2.0}}),
        resolution_m=2.0,
    )["snow_surface_dem"]
    high = compute_snow_surface_arrays(
        dem,
        slope,
        tpi,
        aspect,
        resolve_snow_surface_config({"snow_surface": {**base_cfg, "base_snow_height_m": 6.0}}),
        resolution_m=2.0,
    )["snow_surface_dem"]

    assert np.isclose((high - low).mean(), 4.0, atol=0.75)


def test_slope_leveling_fills_depressions_on_gentle_terrain_only() -> None:
    width = 128
    grid_x = np.linspace(0, 14 * np.pi, width, dtype=np.float32)
    grid_y = np.linspace(0, 14 * np.pi, width, dtype=np.float32)
    dem = np.full((width, width), 2000.0, dtype=np.float32)
    dem += np.sin(grid_x)[None, :] * np.sin(grid_y)[:, None] * 5.0
    slope = np.full((width, width), 12.0, dtype=np.float32)
    slope[:, width // 2 :] = 42.0
    tpi = np.zeros((width, width), dtype=np.float32)
    aspect = np.zeros((width, width), dtype=np.float32)

    cfg = resolve_snow_surface_config(
        {
            "snow_surface": {
                "base_snow_height_m": 2.0,
                "max_accumulation_slope_deg": 35.0,
                "leveling_full_slope_deg": 30.0,
                "leveling_end_slope_deg": 35.0,
                "accumulation_blend_sigma_m": 10.0,
                "smoothing_sigma_m": 40.0,
                "micro_suppression": 0.90,
                "depression_fill": 0.95,
            }
        }
    )
    result = compute_snow_surface_arrays(
        dem, slope, tpi, aspect, cfg, resolution_m=2.0
    )
    gentle_mask = slope < 30.0
    steep_mask = slope >= 35.0
    gentle_thickness = float(result["snow_thickness_m"][gentle_mask].mean())

    assert gentle_thickness > 1.5
    assert float(result["snow_thickness_m"][steep_mask].mean()) < 0.2


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

    uniform = resolve_snow_surface_config(
        {"snow_surface": {"valley_deposition_factor": 0.0, "ridge_scour_factor": 0.0}}
    )
    modulated = resolve_snow_surface_config(
        {
            "snow_surface": {
                "valley_deposition_factor": 0.30,
                "ridge_scour_factor": 0.50,
            }
        }
    )
    tpi_work = np.zeros((height, width), dtype=np.float32)
    tpi_work[20:30, 20:30] = -2.0
    tpi_work[40:45, 40:50] = 2.0
    uniform_surface = compute_snow_surface_arrays(
        dem, slope, tpi_work, aspect, uniform, resolution_m=1.0
    )["snow_thickness_m"]
    modulated_surface = compute_snow_surface_arrays(
        dem, slope, tpi_work, aspect, modulated, resolution_m=1.0
    )["snow_thickness_m"]

    assert modulated_surface.std() > uniform_surface.std()
    assert modulated_surface[25, 25] > uniform_surface[25, 25]
    assert modulated_surface[42, 45] < uniform_surface[42, 45]


def test_leveled_blanket_buries_micro_protrusions_on_gentle_terrain() -> None:
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
                "leveling_full_slope_deg": 30.0,
                "leveling_end_slope_deg": 35.0,
            }
        }
    )
    result = compute_snow_surface_arrays(
        dem, slope, tpi, aspect, cfg, resolution_m=0.5
    )

    assert (result["snow_thickness_m"] >= 0).all()
    assert (result["snow_surface_dem"] >= dem).all()
    assert float(np.std(result["snow_surface_dem"])) <= float(np.std(dem)) * 1.02


def test_ridge_transition_follows_dem_without_edge_dip() -> None:
    """Snow deck should taper above summer relief, not dip before a rising ridge."""
    width = 160
    dem = np.full((width, width), 2000.0, dtype=np.float32)
    for col in range(width):
        if col > width // 2 - 20:
            dem[:, col] += (col - (width // 2 - 20)) * 0.25
    dem[:, : width // 2] += np.sin(np.linspace(0, 24 * np.pi, width // 2)) * 5.0

    slope = np.full((width, width), 12.0, dtype=np.float32)
    slope[:, width // 2 + 5 :] = 50.0
    tpi = np.zeros((width, width), dtype=np.float32)
    aspect = np.zeros((width, width), dtype=np.float32)

    cfg = resolve_snow_surface_config(
        {
            "snow_surface": {
                "base_snow_height_m": 2.0,
                "snow_amount": 0.7,
                "max_accumulation_slope_deg": 35.0,
                "accumulation_transition_deg": 20.0,
                "accumulation_edge_feather_m": 45.0,
                "leveling_full_slope_deg": 30.0,
                "leveling_end_slope_deg": 40.0,
                "smoothing_sigma_m": 90.0,
                "micro_suppression": 0.75,
                "cover_transition_sigma_m": 42.0,
            }
        }
    )
    result = compute_snow_surface_arrays(
        dem, slope, tpi, aspect, cfg, resolution_m=2.0
    )
    snow = result["snow_surface_dem"]
    cover = result["snow_cover_weight"]
    row = width // 2

    transition = (cover[row] > 0.15) & (cover[row] < 0.95)
    assert transition.any()
    offsets = snow[row, transition] - dem[row, transition]
    assert offsets.min() > 0.05

    rising = transition & (np.gradient(dem[row]) > 0.02)
    if rising.any():
        rising_cols = np.where(rising)[0]
        for idx in range(1, len(rising_cols)):
            col = rising_cols[idx]
            prev_col = rising_cols[idx - 1]
            if dem[row, col] >= dem[row, prev_col] - 0.05:
                assert snow[row, col] >= snow[row, prev_col] - 0.2


def test_cliffs_follow_dem_when_leveling_inactive() -> None:
    width = 64
    dem = np.full((width, width), 2000.0, dtype=np.float32)
    dem[20:30, 20:30] += 6.0
    slope = np.full((width, width), 42.0, dtype=np.float32)
    slope[5:15, 5:15] = 10.0
    tpi = np.zeros((width, width), dtype=np.float32)
    aspect = np.zeros((width, width), dtype=np.float32)

    cfg = resolve_snow_surface_config(
        {
            "snow_surface": {
                "base_snow_height_m": 2.0,
                "max_accumulation_slope_deg": 35.0,
                "leveling_full_slope_deg": 30.0,
                "leveling_end_slope_deg": 35.0,
                "micro_suppression": 0.90,
                "smoothing_sigma_m": 40.0,
            }
        }
    )
    result = compute_snow_surface_arrays(
        dem, slope, tpi, aspect, cfg, resolution_m=2.0
    )
    steep = slope >= 35.0
    assert np.allclose(result["snow_surface_dem"][steep], dem[steep], atol=0.05)


def test_peak_retention_zero_produces_smoother_base_than_legacy() -> None:
    from winter_ortho.snow_model import surface as surf

    dem, slope, tpi, aspect = _flat_terrain(width=128, height=128)
    legacy = resolve_snow_surface_config({"snow_surface": {"peak_retention": 1.0}})
    smooth = resolve_snow_surface_config({"snow_surface": {"peak_retention": 0.0}})
    legacy_base = surf._compute_snow_base(dem, legacy, resolution_m=1.0)
    smooth_base = surf._compute_snow_base(dem, smooth, resolution_m=1.0)
    assert float(np.ptp(smooth_base)) < float(np.ptp(legacy_base))


def test_leveled_blanket_cap_prevents_pre_cliff_bulge() -> None:
    """Macro-smoothed blanket must cap ridge bulges but keep depression fill on gentle slopes."""
    width = 160
    dem = np.full((width, width), 2000.0, dtype=np.float32)
    for col in range(width):
        if col > width // 2 - 20:
            dem[:, col] += (col - (width // 2 - 20)) * 0.35
    dem[:, : width // 2] += np.sin(np.linspace(0, 24 * np.pi, width // 2)) * 3.0

    slope = np.full((width, width), 18.0, dtype=np.float32)
    slope[:, width // 2 + 8 :] = 52.0
    tpi = np.zeros((width, width), dtype=np.float32)
    aspect = np.zeros((width, width), dtype=np.float32)

    cfg = resolve_snow_surface_config(
        {
            "snow_surface": {
                "base_snow_height_m": 10.0,
                "max_accumulation_slope_deg": 35.0,
                "accumulation_transition_deg": 20.0,
                "leveling_full_slope_deg": 30.0,
                "leveling_end_slope_deg": 40.0,
                "smoothing_sigma_m": 90.0,
                "micro_suppression": 0.90,
                "cover_transition_sigma_m": 42.0,
            }
        }
    )
    result = compute_snow_surface_arrays(
        dem, slope, tpi, aspect, cfg, resolution_m=2.0
    )
    snow = result["snow_surface_dem"]
    offset = snow - dem
    gentle = slope < 32.0
    pre_cliff = gentle & (np.arange(width)[None, :] < width // 2)

    assert pre_cliff.any()
    dem_local = dem - ndimage.gaussian_filter(dem.astype(np.float64), sigma=4)
    snow_local = snow - ndimage.gaussian_filter(snow.astype(np.float64), sigma=4)
    assert float(snow_local[pre_cliff].std()) < float(dem_local[pre_cliff].std())
    assert float(offset[pre_cliff].max()) <= float(result["snow_thickness_m"][pre_cliff].max()) + 1.0


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
        cover_weight=np.ones((8, 8), dtype=np.float32),
    )
    assert np.allclose(blended, 0.8)

    slope_steep = np.full((8, 8), 50.0, dtype=np.float32)
    blended_steep = blend_hillshade_for_snow(
        base,
        snow,
        fraction,
        slope_steep,
        max_accumulation_slope_deg=30.0,
        cover_weight=np.zeros((8, 8), dtype=np.float32),
    )
    assert np.allclose(blended_steep, 0.2)

    cover_mid = np.full((8, 8), 0.5, dtype=np.float32)
    blended_mid = blend_hillshade_for_snow(
        base,
        snow,
        fraction,
        slope,
        max_accumulation_slope_deg=30.0,
        cover_weight=cover_mid,
    )
    assert np.allclose(blended_mid, 0.5)


def test_blend_hillshade_deck_floor_ramps_with_cover() -> None:
    base = np.full((8, 8), 0.1, dtype=np.float32)
    snow = np.full((8, 8), 0.9, dtype=np.float32)
    fraction = np.full((8, 8), 1.0, dtype=np.float32)
    slope = np.zeros((8, 8), dtype=np.float32)

    cliff = blend_hillshade_for_snow(
        base,
        snow,
        fraction,
        slope,
        max_accumulation_slope_deg=30.0,
        cover_weight=np.zeros((8, 8), dtype=np.float32),
        deck_weight_floor=0.75,
    )
    assert np.allclose(cliff, 0.1)

    flat = blend_hillshade_for_snow(
        base,
        snow,
        fraction,
        slope,
        max_accumulation_slope_deg=30.0,
        cover_weight=np.ones((8, 8), dtype=np.float32),
        deck_weight_floor=0.75,
    )
    assert flat.mean() > 0.85


def test_geometry_cover_uses_horizontal_edge_feather() -> None:
    """3D snow deck must taper over cliff feather distance, not only per-pixel slope."""
    width = 200
    dem = np.full((32, width), 2000.0, dtype=np.float32)
    slope = np.full((32, width), 18.0, dtype=np.float32)
    slope[:, width // 2 :] = 52.0
    tpi = np.zeros((32, width), dtype=np.float32)
    aspect = np.zeros((32, width), dtype=np.float32)
    base_cfg = {
        "base_snow_height_m": 8.0,
        "max_accumulation_slope_deg": 35.0,
        "accumulation_transition_deg": 20.0,
        "leveling_full_slope_deg": 30.0,
        "leveling_end_slope_deg": 45.0,
        "smoothing_sigma_m": 40.0,
        "micro_suppression": 0.85,
    }
    with_feather = resolve_snow_surface_config(
        {"snow_surface": {**base_cfg, "accumulation_edge_feather_m": 45.0}}
    )
    without_feather = resolve_snow_surface_config(
        {"snow_surface": {**base_cfg, "accumulation_edge_feather_m": 0.0}}
    )
    feathered = compute_snow_surface_arrays(
        dem, slope, tpi, aspect, with_feather, resolution_m=1.0
    )
    plain = compute_snow_surface_arrays(
        dem, slope, tpi, aspect, without_feather, resolution_m=1.0
    )
    row = 16
    offset_f = feathered["snow_surface_dem"][row] - dem[row]
    offset_p = plain["snow_surface_dem"][row] - dem[row]
    mid_f = (offset_f > 1.0) & (offset_f < 7.5)
    mid_p = (offset_p > 1.0) & (offset_p < 7.5)
    assert mid_f.sum() > mid_p.sum()
    assert mid_f.sum() >= 12


def test_soften_snow_surface_transitions_reduces_mid_cover_roughness() -> None:
    from winter_ortho.snow_model.surface import _soften_snow_surface_transitions

    snow = np.zeros((64, 64), dtype=np.float32)
    snow += np.sin(np.linspace(0, 40 * np.pi, 64, dtype=np.float32))[None, :] * 2.0
    snow += 2010.0
    cover = np.zeros((64, 64), dtype=np.float32)
    cover[:, 16:48] = 0.5
    cfg = {"surface_transition_smooth_sigma_m": 20.0}
    softened = _soften_snow_surface_transitions(snow, cover, cfg, resolution_m=1.0)
    band = cover > 0.25
    assert float(softened[band].std()) < float(snow[band].std())


def _sample_gpx_line(
    gpx_path: Path,
    *,
    samples: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a GPX track segment in LV95 (easting, northing, gpx_ele)."""
    import xml.etree.ElementTree as ET

    from pyproj import Transformer

    gpx_ns = "http://www.topografix.com/GPX/1/1"
    root = ET.parse(gpx_path).getroot()
    wgs_points: list[tuple[float, float, float]] = []
    for trkpt in root.iter():
        if not trkpt.tag.endswith("trkpt"):
            continue
        lon = float(trkpt.get("lon", "nan"))
        lat = float(trkpt.get("lat", "nan"))
        ele_el = trkpt.find(f"{{{gpx_ns}}}ele")
        ele = float(ele_el.text) if ele_el is not None and ele_el.text else 0.0
        wgs_points.append((lon, lat, ele))
    if len(wgs_points) < 2:
        raise ValueError(f"GPX needs at least two points: {gpx_path}")

    lon0, lat0, _ = wgs_points[0]
    lon1, lat1, _ = wgs_points[-1]
    lons = np.linspace(lon0, lon1, samples)
    lats = np.linspace(lat0, lat1, samples)
    to_lv95 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    eastings, northings = to_lv95.transform(lons, lats)
    return eastings.astype(np.float64), northings.astype(np.float64), lons


def _profile_gpx_snow_offset(
    gpx: Path,
    *,
    samples: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample DEM, snow surface, and slope along a GPX line."""
    import rasterio

    from winter_ortho.utils.config import load_config
    from winter_ortho.utils.paths import tile_paths
    from winter_ortho.viewer.export import _sample_dem_elevation

    region_cfg = load_config("config/regions/finsteraarhorn.yaml")
    paths = tile_paths(region_cfg, "finsteraarhorn_001")
    dem_path = paths.dem
    snow_path = paths.snow_surface_dem
    terrain_path = paths.terrain_features
    if not all(p.exists() for p in (gpx, dem_path, snow_path, terrain_path)):
        pytest.skip(f"finsteraarhorn {gpx.name} fixture not available")

    eastings, northings, _ = _sample_gpx_line(gpx, samples=samples)

    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
        transform = dem_src.transform
    with rasterio.open(snow_path) as snow_src:
        snow = snow_src.read(1)
    with rasterio.open(terrain_path) as terrain_src:
        slope = terrain_src.read(2)

    profile_dem: list[float] = []
    profile_snow: list[float] = []
    profile_slope: list[float] = []
    dist = [0.0]
    for i, (e, n) in enumerate(zip(eastings, northings)):
        if i:
            dist.append(
                dist[-1]
                + float(np.hypot(eastings[i] - eastings[i - 1], northings[i] - northings[i - 1]))
            )
        dem_v = _sample_dem_elevation(dem, transform, e, n)
        if dem_v is None:
            continue
        profile_dem.append(dem_v)
        profile_snow.append(_sample_dem_elevation(snow, transform, e, n))
        profile_slope.append(_sample_dem_elevation(slope, transform, e, n))

    dem_arr = np.asarray(profile_dem, dtype=np.float32)
    snow_arr = np.asarray(profile_snow, dtype=np.float32)
    slope_arr = np.asarray(profile_slope, dtype=np.float32)
    dist_arr = np.asarray(dist[: len(dem_arr)], dtype=np.float64)
    return dist_arr, snow_arr - dem_arr, slope_arr


def test_finsti2_gpx_profile_homogeneous_transition() -> None:
    """Finsteraarhorn flat cut to cliff along data/sample/finsti2.gpx."""
    dist_arr, offset, slope_arr = _profile_gpx_snow_offset(Path("data/sample/finsti2.gpx"))

    assert (offset >= -0.05).all()

    flat = (dist_arr < 170.0) & (slope_arr < 32.0)
    assert flat.sum() >= 20
    assert float(offset[flat].std()) < 2.5
    assert float(offset[flat].max() - offset[flat].min()) < 8.0
    assert float(offset[flat].mean()) > 8.0

    steep = slope_arr >= 50.0
    assert steep.any()
    assert float(offset[steep].mean()) < 1.0

    gentle_pre_steep = (slope_arr < 35.0) & (dist_arr > 170.0) & (dist_arr < 280.0)
    assert gentle_pre_steep.any()
    assert float(offset[gentle_pre_steep].min()) > 1.0
    assert float(offset[gentle_pre_steep].std()) < 4.0
    # Offset should taper with cover before the cliff, not stay at full nominal depth.
    pre_cliff = (dist_arr > 200.0) & (dist_arr < 250.0) & (slope_arr < 35.0)
    assert pre_cliff.any()
    assert float(offset[pre_cliff].max()) < float(offset[flat].mean()) - 1.0


def test_cap_leveled_blanket_ignores_sub_metre_micro_bumps() -> None:
    """Small DEM ripples on flat terrain should not pull the leveled deck up."""
    from winter_ortho.snow_model.surface import _cap_leveled_blanket

    dem = np.full((32, 32), 2000.0, dtype=np.float32)
    dem += np.sin(np.linspace(0, 12 * np.pi, 32, dtype=np.float32))[None, :] * 0.8
    thickness = np.full((32, 32), 8.0, dtype=np.float32)
    leveling_weight = np.ones((32, 32), dtype=np.float32)
    ground_ref = ndimage.gaussian_filter(dem.astype(np.float64), sigma=6).astype(np.float32)
    leveled = (ground_ref + thickness).astype(np.float32)

    capped = _cap_leveled_blanket(
        leveled, dem, thickness, leveling_weight, ground_reference=ground_ref
    )
    assert float(np.std(capped - ground_ref - thickness)) < 0.25


def test_cap_leveled_blanket_floors_micro_suppressed_highs() -> None:
    """Steep nunatak pixels may follow dem+thickness; gentle deck stays on ground ref."""
    from winter_ortho.snow_model.surface import _cap_leveled_blanket

    dem = np.full((32, 32), 2000.0, dtype=np.float32)
    dem[16, 16] = 2010.0
    thickness = np.full((32, 32), 10.0, dtype=np.float32)
    leveling_weight = np.full((32, 32), 1.0, dtype=np.float32)
    leveling_weight[16, 16] = 0.15
    ground_ref = dem - 12.0
    leveled = (ground_ref + thickness).astype(np.float32)

    capped = _cap_leveled_blanket(leveled, dem, thickness, leveling_weight, ground_reference=ground_ref)
    assert capped[16, 16] >= dem[16, 16] + thickness[16, 16] - 0.05
    assert capped[8, 8] <= ground_ref[8, 8] + thickness[8, 8] + 0.05


def test_cap_leveled_blanket_preserves_depression_fill() -> None:
    """Filled depressions must not be capped back down to dem + thickness."""
    from winter_ortho.snow_model.surface import _cap_leveled_blanket

    dem = np.full((32, 32), 2000.0, dtype=np.float32)
    dem[10:22, 10:22] -= 6.0
    thickness = np.full((32, 32), 10.0, dtype=np.float32)
    leveling_weight = np.ones((32, 32), dtype=np.float32)
    ground_ref = np.full((32, 32), 2004.0, dtype=np.float32)
    leveled = np.full((32, 32), 2014.0, dtype=np.float32)

    capped = _cap_leveled_blanket(
        leveled, dem, thickness, leveling_weight, ground_reference=ground_ref
    )
    assert float(capped[16, 16]) >= 2013.5
    assert float(capped.std()) < float(dem.std()) * 0.5


def test_micro_suppressed_high_keeps_full_deck_with_high_cover() -> None:
    width = 96
    dem = np.full((width, width), 3460.0, dtype=np.float32)
    dem[48, 48] = 3465.0
    slope = np.full((width, width), 18.0, dtype=np.float32)
    tpi = np.zeros((width, width), dtype=np.float32)
    aspect = np.zeros((width, width), dtype=np.float32)

    cfg = resolve_snow_surface_config(
        {
            "snow_surface": {
                "base_snow_height_m": 10.0,
                "max_accumulation_slope_deg": 35.0,
                "smoothing_sigma_m": 40.0,
                "micro_suppression": 0.95,
                "leveling_full_slope_deg": 30.0,
                "leveling_end_slope_deg": 40.0,
            }
        }
    )
    result = compute_snow_surface_arrays(dem, slope, tpi, aspect, cfg, resolution_m=1.0)
    offset = result["snow_surface_dem"] - dem
    assert float(offset[40, 48]) > 8.0
    assert float(offset[48, 48]) > 7.5


def test_finsti_gpx_no_snow_holes_on_gentle_deck() -> None:
    """Gentle finsti deck must not drop to bare DEM while cover stays high."""
    import rasterio

    from winter_ortho.utils.config import load_config
    from winter_ortho.utils.paths import tile_paths
    from winter_ortho.viewer.export import _sample_dem_elevation

    dist_arr, offset, slope_arr = _profile_gpx_snow_offset(
        Path("data/sample/finsti.gpx"), samples=500
    )
    region_cfg = load_config("config/regions/finsteraarhorn.yaml")
    paths = tile_paths(region_cfg, "finsteraarhorn_001")
    if not paths.snow_surface_dem.exists():
        pytest.skip("finsteraarhorn snow surface fixture not available")

    eastings, northings, _ = _sample_gpx_line(Path("data/sample/finsti.gpx"), samples=500)
    with rasterio.open(paths.dem) as dem_src:
        dem = dem_src.read(1)
        transform = dem_src.transform

    dem_samples = []
    for e, n in zip(eastings, northings):
        v = _sample_dem_elevation(dem, transform, e, n)
        if v is not None:
            dem_samples.append(v)
    dem_arr = np.asarray(dem_samples[: len(offset)], dtype=np.float32)

    for lo, hi in ((3430, 3500), (3480, 3520)):
        band = (dem_arr >= lo) & (dem_arr <= hi) & (slope_arr < 35.0)
        if not band.any():
            continue
        assert float(offset[band].min()) > 7.0, f"Snow hole below 7 m in {lo}-{hi} m band"
        assert float(offset[band].max() - offset[band].min()) < 3.0


def test_finsti_gpx_steep_high_alpine_tapers_without_render_inflation() -> None:
    """Upper finsti: geometric deck tapers on rock; no phantom 7 m render depth."""
    import rasterio

    from winter_ortho.utils.config import load_config
    from winter_ortho.utils.paths import tile_paths
    from winter_ortho.viewer.export import _sample_dem_elevation

    dist_arr, offset, slope_arr = _profile_gpx_snow_offset(
        Path("data/sample/finsti.gpx"), samples=500
    )
    region_cfg = load_config("config/regions/finsteraarhorn.yaml")
    paths = tile_paths(region_cfg, "finsteraarhorn_001")
    if not paths.dem.exists():
        pytest.skip("finsteraarhorn fixture not available")

    eastings, northings, _ = _sample_gpx_line(Path("data/sample/finsti.gpx"), samples=500)
    with rasterio.open(paths.dem) as dem_src:
        dem = dem_src.read(1)
        transform = dem_src.transform

    dem_samples = []
    for e, n in zip(eastings, northings):
        v = _sample_dem_elevation(dem, transform, e, n)
        if v is not None:
            dem_samples.append(v)
    dem_arr = np.asarray(dem_samples[: len(offset)], dtype=np.float32)

    alpine = (dem_arr >= 3850) & (dem_arr <= 4020)
    assert alpine.any()
    assert float(offset[alpine].max()) < 1.0
    # Taper should be gradual below the rock band, not a late step up.
    taper = (dem_arr >= 3580) & (dem_arr < 3850) & (slope_arr < 45.0)
    if taper.any():
        assert float(offset[taper].max() - offset[taper].min()) < 8.0


def test_finsti_gpx_pre_cliff_tapers_before_rock() -> None:
    """Full finsti route: snow deck thins approaching rock, no ~10 m plateau."""
    import rasterio

    from winter_ortho.utils.config import load_config
    from winter_ortho.utils.paths import tile_paths
    from winter_ortho.viewer.export import _sample_dem_elevation

    dist_arr, offset, slope_arr = _profile_gpx_snow_offset(
        Path("data/sample/finsti.gpx"), samples=500
    )
    region_cfg = load_config("config/regions/finsteraarhorn.yaml")
    paths = tile_paths(region_cfg, "finsteraarhorn_001")
    eastings, northings, _ = _sample_gpx_line(Path("data/sample/finsti.gpx"), samples=500)
    with rasterio.open(paths.dem) as dem_src:
        dem = dem_src.read(1)
        transform = dem_src.transform

    dem_samples = []
    for e, n in zip(eastings, northings):
        v = _sample_dem_elevation(dem, transform, e, n)
        if v is not None:
            dem_samples.append(v)
    dem_arr = np.asarray(dem_samples[: len(offset)], dtype=np.float32)

    steep = slope_arr >= 55.0
    gentle = slope_arr < 30.0
    assert gentle.any() and steep.any()
    assert float(offset[gentle].mean()) > float(offset[steep].mean()) + 3.0

    alpine_taper = (dem_arr >= 3580) & (dem_arr <= 3720) & (slope_arr < 45.0)
    assert alpine_taper.any()
    assert float(offset[alpine_taper].max() - offset[alpine_taper].min()) > 3.0
    assert float(offset[alpine_taper].min()) < 1.5


def test_finsti3_gpx_profile_no_pre_cliff_bulge() -> None:
    """Finsteraarhorn traverse (finsti3) without snow bulges before rock steps."""
    dist_arr, offset, slope_arr = _profile_gpx_snow_offset(Path("data/sample/finsti3.gpx"))

    assert (offset >= -0.05).all()
    assert dist_arr[-1] > 1000.0

    opening = (dist_arr < 400.0) & (slope_arr < 32.0)
    assert opening.sum() >= 15
    assert float(offset[opening].std()) < 1.0
    assert float(offset[opening].max() - offset[opening].min()) < 2.55
    assert float(offset[opening].mean()) > 8.0

    steep = (slope_arr >= 55.0) & (dist_arr > 500.0) & (dist_arr < 1100.0)
    assert steep.any()
    assert float(offset[steep].mean()) < 3.5

    closing = (dist_arr > 1250.0) & (dist_arr < 1620.0) & (slope_arr < 32.0)
    assert closing.any()
    assert float(offset[closing].std()) < 3.0
    assert float(offset[closing].max() - offset[closing].min()) < 8.0

    gentle = slope_arr < 35.0
    assert float(offset[gentle].max()) < 12.0


def test_finsti3_gpx_upper_section_no_center_snow_furrow() -> None:
    """Upper finsti3 traverse: track should not sit in a snow-free furrow beside banks."""
    import rasterio

    from winter_ortho.utils.config import load_config
    from winter_ortho.utils.paths import tile_paths
    from winter_ortho.viewer.export import _sample_dem_elevation

    dist_arr, offset, slope_arr = _profile_gpx_snow_offset(
        Path("data/sample/finsti3.gpx"), samples=500
    )
    region_cfg = load_config("config/regions/finsteraarhorn.yaml")
    paths = tile_paths(region_cfg, "finsteraarhorn_001")
    eastings, northings, _ = _sample_gpx_line(Path("data/sample/finsti3.gpx"), samples=500)

    with rasterio.open(paths.dem) as dem_src:
        dem = dem_src.read(1)
        transform = dem_src.transform
    with rasterio.open(paths.snow_surface_dem) as snow_src:
        snow = snow_src.read(1)

    furrow = (dist_arr > 1450.0) & (dist_arr < 1560.0) & (slope_arr < 35.0)
    assert furrow.sum() >= 8
    assert float(offset[furrow].mean()) > 4.0
    assert float(offset[furrow].min()) > 3.0

    for td in (1483.0, 1536.0):
        i = int(np.argmin(np.abs(dist_arr - td)))
        e0, n0 = eastings[i], northings[i]
        de = eastings[min(i + 1, len(eastings) - 1)] - eastings[max(i - 1, 0)]
        dn = northings[min(i + 1, len(northings) - 1)] - northings[max(i - 1, 0)]
        length = float(np.hypot(de, dn))
        pe, pn = -dn / length, de / length
        lateral_offsets: list[float] = []
        for pm in (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0):
            e = e0 + pe * pm
            n = n0 + pn * pm
            dem_v = _sample_dem_elevation(dem, transform, e, n)
            snow_v = _sample_dem_elevation(snow, transform, e, n)
            assert dem_v is not None and snow_v is not None
            lateral_offsets.append(snow_v - dem_v)
        center = lateral_offsets[3]
        edge = (lateral_offsets[0] + lateral_offsets[1] + lateral_offsets[5] + lateral_offsets[6]) / 4.0
        assert edge - center < 3.0


def test_finsti4_gpx_profile_small_buckle_smoothed() -> None:
    """Short finsti4 segment: no inflated deck above nominal height on gentle slopes."""
    dist_arr, offset, slope_arr = _profile_gpx_snow_offset(Path("data/sample/finsti4.gpx"))

    assert (offset >= -0.05).all()
    assert dist_arr[-1] < 250.0

    gentle = slope_arr < 35.0
    assert gentle.sum() >= 10
    assert float(offset[gentle].max()) < 10.5
    assert float(offset[gentle].std()) < 3.0

    # Taper toward rock at end without a late bulge on the track.
    tail = dist_arr > 100.0
    assert tail.any()
    assert float(offset[tail].max()) < float(offset[gentle].max()) + 0.5


def test_composite_deck_depth_does_not_inflate_above_leveled_blanket() -> None:
    """deck_depth taper must not exceed the leveled deck height above DEM."""
    width = 128
    dem = np.full((width, width), 2000.0, dtype=np.float32)
    dem[:, width // 2 :] += np.linspace(0, 40, width // 2, dtype=np.float32)
    ground_ref = dem - 8.0
    leveled = dem + 10.0
    thickness = np.full((width, width), 10.0, dtype=np.float32)
    cover = np.full((width, width), 0.8, dtype=np.float32)

    from winter_ortho.snow_model.surface import _composite_snow_surface

    snow = _composite_snow_surface(dem, leveled, ground_ref, thickness, cover)
    assert (snow <= leveled + 0.05).all()
    assert (snow >= dem - 0.05).all()


def test_sentischhorn_gpx_2140m_deck_without_offset_kink() -> None:
    """Along Sentischhorn GPX near 2140–2150 m the snow deck must not buckle."""
    import xml.etree.ElementTree as ET

    import rasterio
    from pyproj import Transformer

    from winter_ortho.utils.config import load_config
    from winter_ortho.utils.paths import tile_paths
    from winter_ortho.viewer.export import _sample_dem_elevation
    from winter_ortho.snow_model.surface import compute_snow_surface_arrays, resolve_snow_surface_config

    gpx = Path("data/sample/sentischhorn.gpx")
    if not gpx.exists():
        pytest.skip("sentischhorn GPX not available")

    region_cfg = load_config("config/regions/sentischhorn.yaml")
    profile = load_config("config/rendering_profiles/sentischhorn.yaml")
    paths = tile_paths(region_cfg, "sentischhorn_001")
    if not paths.dem.exists():
        pytest.skip("sentischhorn tile data not available")

    gpx_ns = "http://www.topografix.com/GPX/1/1"
    to_lv95 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    track: list[tuple[float, float, float, float]] = []
    dist0 = 0.0
    wgs_points: list[tuple[float, float, float]] = []
    for trkpt in ET.parse(gpx).getroot().iter():
        if not trkpt.tag.endswith("trkpt"):
            continue
        ele_el = trkpt.find(f"{{{gpx_ns}}}ele")
        wgs_points.append(
            (
                float(trkpt.get("lon")),
                float(trkpt.get("lat")),
                float(ele_el.text) if ele_el is not None and ele_el.text else 0.0,
            )
        )
    lv95 = [to_lv95.transform(lon, lat) + (ele,) for lon, lat, ele in wgs_points]
    for i in range(len(lv95) - 1):
        e0, n0, el0 = lv95[i]
        e1, n1, el1 = lv95[i + 1]
        seg = float(np.hypot(e1 - e0, n1 - n0))
        for t in np.linspace(0, 1, max(2, int(seg / 3)), endpoint=(i == len(lv95) - 2)):
            track.append((e0 + (e1 - e0) * t, n0 + (n1 - n0) * t, el0 + (el1 - el0) * t, dist0 + seg * t))
        dist0 += seg

    with rasterio.open(paths.dem) as dem_src:
        dem = dem_src.read(1)
        transform = dem_src.transform
    with rasterio.open(paths.terrain_features) as terrain_src:
        slope = terrain_src.read(2)
        tpi = terrain_src.read(3)
        aspect = terrain_src.read(4)

    snow = compute_snow_surface_arrays(
        dem,
        slope,
        tpi,
        aspect,
        resolve_snow_surface_config(profile),
        resolution_m=float(region_cfg.get("resolution_m", 1.0)),
    )["snow_surface_dem"]

    offsets: list[float] = []
    for e, n, gpx_ele, _ in track:
        if not (2138.0 <= gpx_ele <= 2152.0):
            continue
        dem_v = _sample_dem_elevation(dem, transform, e, n)
        snow_v = _sample_dem_elevation(snow, transform, e, n)
        if dem_v is None or snow_v is None:
            continue
        offsets.append(float(snow_v - dem_v))

    assert len(offsets) >= 12
    offset_arr = np.asarray(offsets, dtype=np.float32)
    offset_grad = np.gradient(offset_arr)
    offset_curvature = np.gradient(offset_grad)
    assert float(np.abs(offset_grad).max()) < 0.65
    assert float(np.abs(offset_curvature).max()) < 0.45
    assert float(np.ptp(offset_arr)) < 5.0
    assert float(offset_arr.mean()) > 7.0
