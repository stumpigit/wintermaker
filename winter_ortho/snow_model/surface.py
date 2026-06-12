from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from winter_ortho.preprocessing.tiling import get_tile_grid
from winter_ortho.utils.paths import TilePaths
from winter_ortho.utils.progress import PipelineProgress
from winter_ortho.utils.raster import write_cog

DEFAULT_SNOW_SURFACE_CONFIG: dict[str, float] = {
    "base_snow_height_m": 2.0,
    "snow_amount": 1.0,
    "max_accumulation_slope_deg": 30.0,
    "accumulation_transition_deg": 10.0,
    "accumulation_edge_feather_m": 25.0,
    "smoothing_sigma_m": 20.0,
    "micro_suppression": 0.0,
    "depression_fill": 0.85,
    "ridge_micro_retention": 0.25,
    "peak_retention": 1.0,
    "tpi_smoothing_sigma_m": 0.0,
    "thickness_smoothing_sigma_m": 0.0,
    "surface_post_smooth_sigma_m": 0.0,
    "surface_macro_smooth_sigma_m": 0.0,
    "accumulation_blend_sigma_m": 0.0,
    "valley_deposition_factor": 0.30,
    "ridge_scour_factor": 0.50,
    "windward_aspect_penalty": 0.15,
    "leveling_full_slope_deg": 30.0,
    "leveling_end_slope_deg": 0.0,
    "cover_transition_sigma_m": 0.0,
}


def resolve_snow_surface_config(
    profile: dict[str, Any],
    *,
    snow_height_m: float | None = None,
) -> dict[str, float]:
    cfg = {**DEFAULT_SNOW_SURFACE_CONFIG, **profile.get("snow_surface", {})}
    if snow_height_m is not None:
        cfg["base_snow_height_m"] = float(snow_height_m)
    return {key: float(cfg[key]) for key in DEFAULT_SNOW_SURFACE_CONFIG}


def compute_snow_surface_arrays(
    dem: np.ndarray,
    slope: np.ndarray,
    tpi: np.ndarray,
    aspect: np.ndarray,
    cfg: dict[str, float],
    *,
    resolution_m: float,
) -> dict[str, np.ndarray]:
    """Derive snow-covered surface and per-pixel thickness from terrain."""
    base_height = cfg["base_snow_height_m"]
    max_slope = cfg["max_accumulation_slope_deg"]

    slope_work = _smooth_slope_for_weight(slope, cfg, resolution_m)
    leveling_weight = _slope_leveling_weight(slope_work, cfg)
    blend_weight = _resolve_blend_weight(slope, cfg, resolution_m, slope_work=slope_work)
    ground_reference = _ground_reference_surface(
        dem,
        cfg,
        resolution_m,
        leveling_weight=leveling_weight,
    )

    tpi_work = tpi.astype(np.float64)
    tpi_sigma_m = float(cfg.get("tpi_smoothing_sigma_m", 0.0))
    if tpi_sigma_m > 0.0:
        tpi_sigma_px = max(1.0, tpi_sigma_m / resolution_m)
        tpi_work = ndimage.gaussian_filter(tpi_work, sigma=tpi_sigma_px)
    tpi_norm = _normalize_tpi(tpi_work.astype(np.float32))
    thickness = np.full(dem.shape, base_height, dtype=np.float32)

    valley_factor = np.clip(-tpi_norm, 0.0, 1.0) * leveling_weight
    thickness *= 1.0 + cfg["valley_deposition_factor"] * valley_factor

    ridge_factor = np.clip(tpi_norm, 0.0, 1.0) * leveling_weight
    thickness *= 1.0 - cfg["ridge_scour_factor"] * ridge_factor

    windward = np.clip(np.cos(np.radians(aspect - 202.5)), 0.0, 1.0)
    thickness *= 1.0 - cfg["windward_aspect_penalty"] * ridge_factor * windward

    thickness = np.maximum(thickness, 0.0).astype(np.float32)

    snow_amount = float(cfg.get("snow_amount", 1.0))
    if snow_amount != 1.0:
        thickness = (thickness * snow_amount).astype(np.float32)

    thickness_sigma_m = float(cfg.get("thickness_smoothing_sigma_m", 0.0))
    if thickness_sigma_m > 0.0:
        thickness_sigma_px = max(1.0, thickness_sigma_m / resolution_m)
        thickness = ndimage.gaussian_filter(
            thickness.astype(np.float64),
            sigma=thickness_sigma_px,
        ).astype(np.float32)

    # Blanket on a leveled ground reference, tapered toward steep faces.
    on_accumulation = slope_work < max_slope
    snow_layer = (thickness * blend_weight).astype(np.float32)
    snow_surface = (ground_reference + snow_layer).astype(np.float32)
    smooth_weight = _surface_smooth_weight(blend_weight, cfg, leveling_weight)
    snow_surface = _apply_surface_smoothing(
        snow_surface,
        smooth_weight,
        cfg,
        resolution_m,
        leveling_weight=leveling_weight,
    )
    leveled_blanket = snow_surface.astype(np.float32)
    cover = (blend_weight * leveling_weight).astype(np.float32)
    cover = _smooth_cover_weight(cover, cfg, resolution_m)
    snow_surface = (
        leveled_blanket * cover + dem.astype(np.float32) * (1.0 - cover)
    ).astype(np.float32)
    snow_thickness = _finalize_snow_thickness(
        dem,
        snow_surface,
        leveled_blanket,
        thickness,
        cover,
    )

    accumulation = on_accumulation
    return {
        "snow_surface_dem": snow_surface,
        "snow_thickness_m": snow_thickness,
        "accumulation_mask": accumulation.astype(np.uint8),
    }


def _smooth_cover_weight(
    cover: np.ndarray,
    cfg: dict[str, float],
    resolution_m: float,
) -> np.ndarray:
    """Spatially soften snow↔rock transitions (single blended ramp, no step bands)."""
    sigma_m = float(cfg.get("cover_transition_sigma_m", 0.0))
    if sigma_m <= 0.0:
        return cover.astype(np.float32)
    sigma_px = max(1.0, sigma_m / resolution_m)
    smoothed = ndimage.gaussian_filter(cover.astype(np.float64), sigma=sigma_px).astype(
        np.float32
    )
    return np.clip(smoothed, 0.0, 1.0).astype(np.float32)


def _finalize_snow_thickness(
    dem: np.ndarray,
    snow_surface: np.ndarray,
    leveled_blanket: np.ndarray,
    thickness: np.ndarray,
    cover: np.ndarray,
) -> np.ndarray:
    """Thickness for layers/QA; leveled terrain keeps nominal depth even when rock is buried."""
    geometric = np.maximum(snow_surface - dem, 0.0).astype(np.float32)
    nominal = (thickness * cover).astype(np.float32)
    on_blanket = cover > 0.5
    return np.where(
        on_blanket,
        np.maximum(nominal, geometric),
        geometric,
    ).astype(np.float32)


def _slope_leveling_weight(
    slope_work: np.ndarray,
    cfg: dict[str, float],
) -> np.ndarray:
    """Per-pixel DEM leveling: 1 on gentle slopes, 0 on steep (fade between thresholds)."""
    full_deg = float(cfg.get("leveling_full_slope_deg", 30.0))
    end_deg = float(cfg.get("leveling_end_slope_deg", 0.0))
    if end_deg <= 0.0:
        end_deg = cfg["max_accumulation_slope_deg"]
    if end_deg <= full_deg:
        end_deg = full_deg + 1e-3
    return (1.0 - _smoothstep(full_deg, end_deg, slope_work.astype(np.float64))).astype(np.float32)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Hermite ramp from 0 (at edge0) to 1 (at edge1)."""
    width = max(edge1 - edge0, 1e-3)
    t = np.clip((x - edge0) / width, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _slope_accumulation_weight(
    slope: np.ndarray,
    *,
    max_slope: float,
    transition_deg: float,
) -> np.ndarray:
    """Per-pixel snow weight: 1 below max_slope, smooth ramp to 0 on steeper faces."""
    lo = max_slope
    hi = max_slope + transition_deg
    return (1.0 - _smoothstep(lo, hi, slope.astype(np.float64))).astype(np.float32)


def _edge_feather_weight(
    slope: np.ndarray,
    cfg: dict[str, float],
    resolution_m: float,
    *,
    slope_work: np.ndarray | None = None,
) -> np.ndarray:
    """Fade snow toward cliff edges by horizontal distance from steep terrain."""
    feather_m = float(cfg.get("accumulation_edge_feather_m", 25.0))
    if feather_m <= 0.0:
        return np.ones_like(slope, dtype=np.float32)

    max_slope = cfg["max_accumulation_slope_deg"]
    steep_source = slope_work if slope_work is not None else slope
    steep = steep_source >= max_slope
    if not np.any(steep):
        return np.ones_like(slope, dtype=np.float32)
    if np.all(steep):
        return np.zeros_like(slope, dtype=np.float32)
    dist_px = ndimage.distance_transform_edt(~steep)
    dist_m = dist_px.astype(np.float64) * resolution_m
    return _smoothstep(0.0, feather_m, dist_m)


def _smooth_slope_for_weight(
    slope: np.ndarray,
    cfg: dict[str, float],
    resolution_m: float,
) -> np.ndarray:
    """Reduce pixel-scale slope noise that causes salt-and-pepper snow holes."""
    sigma_m = float(cfg.get("accumulation_blend_sigma_m", 0.0))
    if sigma_m <= 0.0:
        return slope.astype(np.float32)
    sigma_px = max(1.0, sigma_m / resolution_m)
    return ndimage.gaussian_filter(
        slope.astype(np.float64),
        sigma=sigma_px,
    ).astype(np.float32)


def _resolve_slope_weight(
    slope: np.ndarray,
    cfg: dict[str, float],
    resolution_m: float,
    *,
    slope_work: np.ndarray | None = None,
) -> np.ndarray:
    """Snow weight from smoothed slope; raw slope only caps when no blend smoothing is set."""
    max_slope = cfg["max_accumulation_slope_deg"]
    transition_deg = float(cfg.get("accumulation_transition_deg", 10.0))
    if slope_work is None:
        slope_work = _smooth_slope_for_weight(slope, cfg, resolution_m)
    weight = _slope_accumulation_weight(
        slope_work,
        max_slope=max_slope,
        transition_deg=transition_deg,
    )
    blend_sigma_m = float(cfg.get("accumulation_blend_sigma_m", 0.0))
    if blend_sigma_m <= 0.0:
        weight_cap = _slope_accumulation_weight(
            slope,
            max_slope=max_slope,
            transition_deg=transition_deg,
        )
        return np.minimum(weight, weight_cap).astype(np.float32)
    return weight.astype(np.float32)


def _resolve_blend_weight(
    slope: np.ndarray,
    cfg: dict[str, float],
    resolution_m: float,
    *,
    slope_work: np.ndarray | None = None,
) -> np.ndarray:
    """Combine slope taper and horizontal cliff feathering into one snow weight."""
    slope_weight = _resolve_slope_weight(
        slope,
        cfg,
        resolution_m,
        slope_work=slope_work,
    )
    edge_weight = _edge_feather_weight(
        slope,
        cfg,
        resolution_m,
        slope_work=slope_work,
    )
    return (slope_weight * edge_weight).astype(np.float32)


def _ground_reference_surface(
    dem: np.ndarray,
    cfg: dict[str, float],
    resolution_m: float,
    *,
    leveling_weight: np.ndarray,
) -> np.ndarray:
    """Blend raw DEM with leveled snow base; strength from micro_suppression × slope."""
    micro = float(cfg.get("micro_suppression", 0.0))
    if micro <= 0.0:
        return dem.astype(np.float32)
    strength = (leveling_weight * micro).astype(np.float32)
    if not np.any(strength > 1e-6):
        return dem.astype(np.float32)
    snow_base = _compute_snow_base(dem, cfg, resolution_m)
    return (dem * (1.0 - strength) + snow_base * strength).astype(np.float32)


def _compute_snow_base(
    dem: np.ndarray,
    cfg: dict[str, float],
    resolution_m: float,
) -> np.ndarray:
    sigma_m = cfg["smoothing_sigma_m"]
    sigma_px = max(1.0, sigma_m / resolution_m)
    macro = ndimage.gaussian_filter(dem.astype(np.float64), sigma=sigma_px).astype(np.float32)

    micro_suppression = float(cfg.get("micro_suppression", 0.0))
    if micro_suppression > 0.0:
        micro = dem - macro
        depression_fill = float(cfg.get("depression_fill", 0.85))
        ridge_micro_retention = float(cfg.get("ridge_micro_retention", 0.25))
        micro_target = np.where(
            micro < 0.0,
            micro * (1.0 - depression_fill),
            micro * ridge_micro_retention,
        )
        micro_snow = micro + (micro_target - micro) * micro_suppression
        return (macro + micro_snow).astype(np.float32)

    peak_retention = float(cfg.get("peak_retention", 1.0))
    excess = np.maximum(dem - macro, 0.0)
    return (macro + excess * peak_retention).astype(np.float32)


def _surface_smooth_weight(
    blend_weight: np.ndarray,
    cfg: dict[str, float],
    leveling_weight: np.ndarray,
) -> np.ndarray:
    """Post-smooth in transitions and lightly in the interior when leveling is active."""
    transition = 4.0 * blend_weight * (1.0 - blend_weight)
    micro = float(cfg.get("micro_suppression", 0.0))
    interior = blend_weight * leveling_weight * micro * 0.35
    return np.maximum(transition, interior).astype(np.float32)


def _apply_surface_smoothing(
    snow_surface: np.ndarray,
    smooth_weight: np.ndarray,
    cfg: dict[str, float],
    resolution_m: float,
    *,
    leveling_weight: np.ndarray,
) -> np.ndarray:
    blend = np.clip(smooth_weight, 0.0, 1.0).astype(np.float32)
    post_sigma_m = float(cfg.get("surface_post_smooth_sigma_m", 0.0))
    if post_sigma_m > 0.0:
        post_sigma_px = max(1.0, post_sigma_m / resolution_m)
        snow_smooth = ndimage.gaussian_filter(
            snow_surface.astype(np.float64),
            sigma=post_sigma_px,
        ).astype(np.float32)
        snow_surface = (
            snow_surface * (1.0 - blend) + snow_smooth * blend
        ).astype(np.float32)

    macro_sigma_m = float(cfg.get("surface_macro_smooth_sigma_m", 0.0))
    if macro_sigma_m > 0.0:
        macro_sigma_px = max(1.0, macro_sigma_m / resolution_m)
        snow_macro = ndimage.gaussian_filter(
            snow_surface.astype(np.float64),
            sigma=macro_sigma_px,
        ).astype(np.float32)
        macro_blend = (blend * leveling_weight).astype(np.float32)
        snow_surface = (
            snow_surface * (1.0 - macro_blend) + snow_macro * macro_blend
        ).astype(np.float32)

    return snow_surface.astype(np.float32)


def compute_snow_surface(
    config: dict[str, Any],
    profile: dict[str, Any],
    paths: TilePaths,
    terrain: dict[str, np.ndarray],
    *,
    snow_height_m: float | None = None,
    progress: PipelineProgress | None = None,
) -> dict[str, np.ndarray]:
    grid = get_tile_grid(config, paths.tile_id)
    surface_cfg = resolve_snow_surface_config(profile, snow_height_m=snow_height_m)
    resolution_m = float(config.get("resolution_m", 2.0))

    if progress:
        progress.substep(
            f"Computing snow surface (base height {surface_cfg['base_snow_height_m']:.1f} m, "
            f"slope < {surface_cfg['max_accumulation_slope_deg']:.0f}°)"
        )

    arrays = compute_snow_surface_arrays(
        terrain["elevation"],
        terrain["slope"],
        terrain["terrain_position_index"],
        terrain["aspect"],
        surface_cfg,
        resolution_m=resolution_m,
    )

    if progress:
        progress.substep("Writing snow_surface_dem.tif and snow_thickness_m.tif")

    for name in ("snow_surface_dem", "snow_thickness_m"):
        write_cog(
            str(getattr(paths, name)),
            arrays[name],
            transform=grid.transform,
            crs=grid.crs,
            nodata=-9999.0,
        )

    write_cog(
        str(paths.accumulation_mask),
        arrays["accumulation_mask"],
        transform=grid.transform,
        crs=grid.crs,
        nodata=0,
    )

    return arrays


def _normalize_tpi(tpi: np.ndarray) -> np.ndarray:
    finite = tpi[np.isfinite(tpi)]
    if finite.size == 0:
        return np.zeros_like(tpi, dtype=np.float32)
    scale = max(float(np.percentile(np.abs(finite), 90)), 1e-3)
    return np.clip(tpi / scale, -1.0, 1.0).astype(np.float32)
