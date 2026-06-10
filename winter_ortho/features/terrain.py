from __future__ import annotations

import gc
from typing import Any

import numpy as np
import rasterio
from scipy import ndimage

from winter_ortho.preprocessing.tiling import get_tile_grid
from winter_ortho.utils.grid_stats import check_grid_memory, format_grid_summary
from winter_ortho.utils.paths import TilePaths
from winter_ortho.utils.progress import PipelineProgress

FEATURE_NAMES = [
    "elevation",
    "slope",
    "aspect",
    "curvature_plan",
    "curvature_profile",
    "hillshade_winter_low_sun",
    "terrain_position_index",
    "local_relief",
    "roughness",
    "flow_accumulation_proxy",
]

# Bands required downstream (snow model); avoid loading the full stack later
SNOW_TERRAIN_BANDS = [
    "elevation",
    "slope",
    "aspect",
    "terrain_position_index",
    "hillshade_winter_low_sun",
    "roughness",
]


def compute_terrain_features(
    config: dict[str, Any],
    paths: TilePaths,
    *,
    progress: PipelineProgress | None = None,
) -> dict[str, np.ndarray]:
    grid = get_tile_grid(config, paths.tile_id)
    terrain_cfg = config.get("terrain", {})
    resolution_m = float(config["resolution_m"])

    for msg in check_grid_memory(grid):
        if progress:
            progress.warn(msg)

    if progress:
        progress.substep(f"Grid: {format_grid_summary(grid)}")
        progress.substep(f"Reading DEM ({paths.dem.name})")

    with rasterio.open(str(paths.dem)) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0

    valid = dem != nodata
    dem_filled = np.where(valid, dem, 0.0)

    if progress:
        progress.substep("Computing slope, aspect, curvature")

    dz_dy, dz_dx = np.gradient(dem_filled, resolution_m, resolution_m)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))).astype(np.float32)
    aspect = ((np.degrees(np.arctan2(-dz_dy, dz_dx)) + 360) % 360).astype(np.float32)

    d2z_dx2 = ndimage.sobel(dz_dx, axis=1) / (8 * resolution_m)
    d2z_dy2 = ndimage.sobel(dz_dy, axis=0) / (8 * resolution_m)
    curvature_plan = (d2z_dx2 + d2z_dy2).astype(np.float32)

    slope_grad = np.sqrt(dz_dx**2 + dz_dy**2) + 1e-8
    p = dz_dx / slope_grad
    q = dz_dy / slope_grad
    curvature_profile = (
        np.gradient(p, resolution_m, axis=1) + np.gradient(q, resolution_m, axis=0)
    ).astype(np.float32)
    del d2z_dx2, d2z_dy2, p, q, slope_grad

    if progress:
        progress.substep("Computing hillshade, TPI, relief, roughness")

    hillshade = _hillshade_from_gradients(dz_dx, dz_dy, terrain_cfg)
    tpi_radius = int(terrain_cfg.get("tpi_radius_px", 5))
    roughness_radius = int(terrain_cfg.get("roughness_radius_px", 3))
    tpi = _terrain_position_index(dem_filled, dem, valid, tpi_radius)
    roughness = _roughness_fast(dem_filled, roughness_radius)
    local_relief = _local_relief(dem_filled, tpi_radius)

    if progress:
        progress.substep("Computing flow accumulation proxy")

    flow = _flow_accumulation_proxy(
        dz_dx,
        dz_dy,
        iterations=int(terrain_cfg.get("flow_iterations", 3)),
    )
    del dz_dx, dz_dy

    elevation = np.where(valid, dem, 0.0).astype(np.float32)
    del dem, dem_filled, valid
    gc.collect()

    band_arrays = [
        elevation,
        slope,
        aspect,
        curvature_plan,
        curvature_profile,
        hillshade,
        tpi,
        local_relief,
        roughness,
        flow,
    ]

    if progress:
        progress.substep(
            f"Writing terrain_features.tif ({len(FEATURE_NAMES)} bands, band-by-band)"
        )

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": grid.width,
        "height": grid.height,
        "count": len(FEATURE_NAMES),
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": -9999.0,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }
    with rasterio.open(str(paths.terrain_features), "w", **profile) as dst:
        for idx, arr in enumerate(band_arrays, start=1):
            dst.write(arr.astype(np.float32), idx)

    features = {name: band_arrays[i] for i, name in enumerate(FEATURE_NAMES)}
    # Release arrays not needed by caller (pipeline reloads from disk for snow)
    return {name: features[name] for name in SNOW_TERRAIN_BANDS if name in features}


def load_terrain_bands(
    paths: TilePaths,
    band_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Load only selected terrain bands to save memory."""
    names = band_names or SNOW_TERRAIN_BANDS
    name_to_idx = {name: i + 1 for i, name in enumerate(FEATURE_NAMES)}
    result: dict[str, np.ndarray] = {}
    with rasterio.open(str(paths.terrain_features)) as src:
        for name in names:
            if name not in name_to_idx:
                continue
            result[name] = src.read(name_to_idx[name]).astype(np.float32)
    return result


def hillshade_config_for_render(
    terrain_cfg: dict[str, Any],
    map_shading: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge terrain hillshade settings with per-profile map sun direction."""
    base = dict(terrain_cfg)
    hill = dict(base.get("hillshade", {}))
    if map_shading:
        if "sun_azimuth" in map_shading:
            hill["azimuth"] = float(map_shading["sun_azimuth"])
        if "sun_altitude" in map_shading:
            hill["altitude"] = float(map_shading["sun_altitude"])
    base["hillshade"] = hill
    return base


def compute_generalized_hillshade(
    elevation: np.ndarray,
    resolution_m: float,
    terrain_cfg: dict[str, Any],
) -> np.ndarray:
    """Hillshade from Gaussian-smoothed DEM — mountains/slopes, not metre-scale noise."""
    sigma_m = float(terrain_cfg.get("generalized_hillshade_sigma_m", 55.0))
    sigma_px = max(1.0, sigma_m / resolution_m)
    smooth = ndimage.gaussian_filter(elevation.astype(np.float64), sigma=sigma_px).astype(np.float32)
    dz_dy, dz_dx = np.gradient(smooth, resolution_m, resolution_m)
    return _hillshade_from_gradients(dz_dx, dz_dy, terrain_cfg)


def _hillshade_from_gradients(
    dz_dx: np.ndarray,
    dz_dy: np.ndarray,
    terrain_cfg: dict[str, Any],
) -> np.ndarray:
    hillshade_cfg = terrain_cfg.get("hillshade", {})
    azimuth = float(hillshade_cfg.get("azimuth", 150))
    altitude = float(hillshade_cfg.get("altitude", 25))
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    az_rad = np.radians(360 - azimuth + 90)
    alt_rad = np.radians(altitude)
    shaded = (
        np.sin(alt_rad) * np.cos(slope_rad)
        + np.cos(alt_rad) * np.sin(slope_rad) * np.cos(az_rad - aspect_rad)
    )
    return np.clip(shaded, 0, 1).astype(np.float32)


def _terrain_position_index(
    dem_filled: np.ndarray,
    dem_raw: np.ndarray,
    valid: np.ndarray,
    radius_px: int,
) -> np.ndarray:
    size = radius_px * 2 + 1
    mean = ndimage.uniform_filter(dem_filled, size=size)
    tpi = dem_raw.astype(np.float32) - mean
    tpi[~valid] = 0.0
    return tpi.astype(np.float32)


def _roughness_fast(elevation: np.ndarray, radius_px: int) -> np.ndarray:
    """Vectorized local std — avoids scipy.generic_filter (very slow on large rasters)."""
    size = radius_px * 2 + 1
    mean = ndimage.uniform_filter(elevation, size=size)
    mean_sq = ndimage.uniform_filter(elevation * elevation, size=size)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(variance).astype(np.float32)


def _local_relief(elevation: np.ndarray, radius_px: int) -> np.ndarray:
    size = radius_px * 2 + 1
    return (
        ndimage.maximum_filter(elevation, size=size) - ndimage.minimum_filter(elevation, size=size)
    ).astype(np.float32)


def _flow_accumulation_proxy(
    dz_dx: np.ndarray,
    dz_dy: np.ndarray,
    *,
    iterations: int,
) -> np.ndarray:
    magnitude = np.sqrt(dz_dx**2 + dz_dy**2) + 1e-6
    flow = (1.0 / magnitude).astype(np.float32)
    iterations = max(1, min(iterations, 10))
    for _ in range(iterations):
        flow = ndimage.uniform_filter(flow, size=5)
    peak = float(flow.max()) + 1e-6
    return (flow / peak).astype(np.float32)
