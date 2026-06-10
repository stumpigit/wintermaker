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
    "max_accumulation_slope_deg": 30.0,
    "smoothing_sigma_m": 20.0,
    "valley_deposition_factor": 0.30,
    "ridge_scour_factor": 0.50,
    "windward_aspect_penalty": 0.15,
    "steep_slope_thickness_factor": 0.35,
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
    sigma_m = cfg["smoothing_sigma_m"]

    accumulation = slope < max_slope

    sigma_px = max(1.0, sigma_m / resolution_m)
    smoothed = ndimage.gaussian_filter(dem.astype(np.float64), sigma=sigma_px).astype(np.float32)

    # Small depressions are filled up to the smoothed reference in accumulation zones.
    leveled = np.where(accumulation, np.maximum(dem, smoothed), dem)

    tpi_norm = _normalize_tpi(tpi)
    thickness = np.full(dem.shape, base_height, dtype=np.float32)

    valley_factor = np.clip(-tpi_norm, 0.0, 1.0)
    thickness *= 1.0 + cfg["valley_deposition_factor"] * valley_factor

    ridge_factor = np.clip(tpi_norm, 0.0, 1.0)
    thickness *= 1.0 - cfg["ridge_scour_factor"] * ridge_factor

    windward = np.clip(np.cos(np.radians(aspect - 202.5)), 0.0, 1.0)
    thickness *= 1.0 - cfg["windward_aspect_penalty"] * ridge_factor * windward

    steep_factor = np.clip((slope - max_slope) / max(45.0 - max_slope, 1e-3), 0.0, 1.0)
    steep_retention = cfg["steep_slope_thickness_factor"]
    thickness *= 1.0 - steep_factor * (1.0 - steep_retention)

    thickness = np.maximum(thickness, 0.0).astype(np.float32)
    snow_surface = (leveled + thickness).astype(np.float32)
    snow_thickness = (snow_surface - dem).astype(np.float32)

    return {
        "snow_surface_dem": snow_surface,
        "snow_thickness_m": snow_thickness,
        "accumulation_mask": accumulation.astype(np.uint8),
    }


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
