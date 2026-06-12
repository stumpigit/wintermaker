from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy import ndimage

from winter_ortho.preprocessing.tiling import get_tile_grid
from winter_ortho.snow_model.surface import resolve_snow_surface_config
from winter_ortho.utils.paths import TilePaths
from winter_ortho.utils.progress import PipelineProgress
from winter_ortho.utils.raster import read_raster, write_cog

# Matches config/class_rules.yaml priority (highest first)
SNOW_CLASS_ORDER = [
    "building_mask",
    "road_mask",
    "path_mask",
    "water_mask",
    "settlement_mask",
    "forest_mask",
    "rock_or_bare_ground_mask",
    "open_land_mask",
]


def compute_snow_layers(
    config: dict[str, Any],
    profile: dict[str, Any],
    paths: TilePaths,
    class_masks: dict[str, np.ndarray],
    terrain: dict[str, np.ndarray],
    *,
    snow_thickness: np.ndarray | None = None,
    snow_surface_dem: np.ndarray | None = None,
    progress: PipelineProgress | None = None,
) -> dict[str, np.ndarray]:
    grid = get_tile_grid(config, paths.tile_id)
    height, width = terrain["elevation"].shape

    if progress:
        progress.substep(
            f"Applying snow rules (profile={profile.get('profile', 'unknown')}, "
            f"{height}×{width} px, priority-aware)"
        )

    snow_fraction = np.zeros((height, width), dtype=np.float32)
    snow_brightness = np.full((height, width), 0.9, dtype=np.float32)
    snow_texture_strength = np.zeros((height, width), dtype=np.float32)
    rock_visibility = np.zeros((height, width), dtype=np.float32)
    forest_snow_intensity = np.zeros((height, width), dtype=np.float32)
    road_visibility = np.zeros((height, width), dtype=np.float32)
    roof_snow_intensity = np.zeros((height, width), dtype=np.float32)
    ice_probability = np.zeros((height, width), dtype=np.float32)

    use_thickness = snow_thickness is not None and "snow_surface" in profile
    thickness_fraction = None
    burial_fraction = None
    slope_snow_scale = None
    protrusion_fraction = None
    if use_thickness:
        surface_cfg = resolve_snow_surface_config(profile)
        base_height = max(surface_cfg["base_snow_height_m"], 1e-3)
        thickness_fraction = np.clip(snow_thickness / base_height, 0.0, 1.0).astype(np.float32)
        resolution_m = float(config.get("resolution_m", 2.0))
        rock_cfg = profile.get("rock", {})
        burial_radius_m = float(rock_cfg.get("thickness_burial_radius_m", 20.0))
        burial_fraction = _local_burial_fraction(
            snow_thickness,
            base_height,
            resolution_m=resolution_m,
            radius_m=burial_radius_m,
        )

    open_cfg = profile.get("open_land", {})
    slope = terrain["slope"]
    if use_thickness and snow_surface_dem is not None:
        slope_snow_scale = _open_land_slope_snow_scale(slope, open_cfg)
        protrusion_fraction = _terrain_protrusion_fraction(
            terrain["elevation"],
            snow_surface_dem,
            full_m=float(open_cfg.get("protrusion_full_m", 0.6)),
        )

    summer_exposure = np.zeros((height, width), dtype=np.float32)

    elev_mod = _elevation_modifier(terrain["elevation"], profile.get("elevation", {}))
    aspect_mod = _aspect_modifier(terrain["aspect"], profile.get("aspect", {}))
    tpi_mod = _normalize(terrain["terrain_position_index"])
    hillshade_mod = 1.0 - terrain["hillshade_winter_low_sun"]
    rough = _normalize(terrain["roughness"])

    claimed = np.zeros((height, width), dtype=bool)
    class_masks_bool = {k: v > 0 for k, v in class_masks.items()}

    def apply_exclusive(name: str, fn: Callable[[np.ndarray], None]) -> None:
        nonlocal claimed
        mask = class_masks_bool.get(name)
        if mask is None:
            return
        exclusive = mask & ~claimed
        if exclusive.any():
            fn(exclusive)
        claimed |= mask

    apply_exclusive("building_mask", lambda m: _apply_building(
        m, snow_fraction, roof_snow_intensity, elev_mod, profile
    ))
    apply_exclusive("road_mask", lambda m: _apply_road(
        m, snow_fraction, road_visibility, profile["roads"]
    ))
    apply_exclusive("path_mask", lambda m: _apply_paths(
        m, snow_fraction, snow_brightness, snow_texture_strength, profile["paths"]
    ))
    apply_exclusive("water_mask", lambda m: _apply_water(
        m, snow_fraction, ice_probability, profile["water"]
    ))
    apply_exclusive("settlement_mask", lambda m: _apply_settlement(
        m, snow_fraction, snow_brightness, snow_texture_strength, elev_mod, profile
    ))
    apply_exclusive("forest_mask", lambda m: _apply_forest(
        m,
        snow_fraction,
        snow_texture_strength,
        forest_snow_intensity,
        elev_mod,
        hillshade_mod,
        profile,
        thickness_fraction=thickness_fraction,
    ))
    apply_exclusive("rock_or_bare_ground_mask", lambda m: _apply_rock(
        m,
        snow_fraction,
        rock_visibility,
        slope,
        rough,
        tpi_mod,
        aspect_mod,
        profile,
        thickness_fraction=thickness_fraction,
        burial_fraction=burial_fraction,
    ))
    apply_exclusive("open_land_mask", lambda m: _apply_open_land(
        m,
        snow_fraction,
        snow_brightness,
        snow_texture_strength,
        summer_exposure,
        elev_mod,
        profile,
        thickness_fraction=thickness_fraction,
        slope_snow_scale=slope_snow_scale,
        protrusion_fraction=protrusion_fraction,
    ))

    shore_width = int(profile["water"].get("shore_snow_width_px", 3))
    if shore_width > 0:
        water_mask = class_masks_bool.get("water_mask", np.zeros_like(claimed))
        if water_mask.any():
            dist = ndimage.distance_transform_edt(~water_mask)
            shore = (dist > 0) & (dist <= shore_width) & ~water_mask
            shore_snow = float(profile["water"].get("shore_snow_intensity", 0.4))
            snow_fraction[shore] = np.maximum(snow_fraction[shore], shore_snow)

    if use_thickness:
        natural_masks = (
            class_masks_bool.get("open_land_mask", np.zeros_like(claimed))
            | class_masks_bool.get("forest_mask", np.zeros_like(claimed))
            | class_masks_bool.get("rock_or_bare_ground_mask", np.zeros_like(claimed))
        )
        natural = claimed & natural_masks
        modulated = snow_fraction.copy()
        modulated[natural] = snow_fraction[natural] + aspect_mod[natural] * 0.05
    else:
        modulated = snow_fraction + aspect_mod * 0.09 + tpi_mod * 0.10
    snow_fraction[:] = np.clip(np.where(claimed, modulated, snow_fraction), 0, 1)

    layers = {
        "snow_fraction": snow_fraction,
        "snow_brightness": snow_brightness,
        "snow_texture_strength": snow_texture_strength,
        "rock_visibility": rock_visibility,
        "forest_snow_intensity": forest_snow_intensity,
        "road_visibility": road_visibility,
        "roof_snow_intensity": roof_snow_intensity,
        "ice_probability": ice_probability,
        "summer_exposure": summer_exposure,
    }

    if progress:
        progress.substep(f"Writing {len(layers)} snow layers to {paths.output_dir}")

    for name, array in layers.items():
        out = getattr(paths, name)
        write_cog(str(out), array.astype(np.float32), transform=grid.transform, crs=grid.crs, nodata=-9999.0)

    labeled, _ = read_raster(str(paths.tlm_masks))
    if labeled.ndim == 3:
        labeled = labeled[0]
    write_cog(
        str(paths.landcover_mask),
        labeled.astype(np.uint8),
        transform=grid.transform,
        crs=grid.crs,
        nodata=0,
    )

    return layers


def _apply_building(
    mask: np.ndarray,
    snow_fraction: np.ndarray,
    roof_snow_intensity: np.ndarray,
    elev_mod: np.ndarray,
    profile: dict[str, Any],
) -> None:
    blo, bhi = profile["buildings"]["roof_snow_intensity"]
    intensity = (blo + bhi) / 2.0 + elev_mod[mask] * 0.2
    roof_snow_intensity[mask] = intensity
    snow_fraction[mask] = intensity


def _apply_road(
    mask: np.ndarray,
    snow_fraction: np.ndarray,
    road_visibility: np.ndarray,
    road_profile: dict[str, Any],
) -> None:
    rlo, rhi = road_profile["snow_fraction"]
    snow_fraction[mask] = (rlo + rhi) / 2.0
    road_visibility[mask] = road_profile["road_visibility"]


def _apply_paths(
    mask: np.ndarray,
    snow_fraction: np.ndarray,
    snow_brightness: np.ndarray,
    snow_texture_strength: np.ndarray,
    path_profile: dict[str, Any],
) -> None:
    lo, hi = path_profile["snow_fraction"]
    snow_fraction[mask] = (lo + hi) / 2.0
    snow_brightness[mask] = path_profile.get("snow_brightness", 0.94)
    snow_texture_strength[mask] = path_profile.get("snow_texture_strength", 0.42)


def _apply_water(
    mask: np.ndarray,
    snow_fraction: np.ndarray,
    ice_probability: np.ndarray,
    water_profile: dict[str, Any],
) -> None:
    snow_fraction[mask] = 0.0
    ice_probability[mask] = water_profile.get("ice_probability", 0.0)


def _apply_settlement(
    mask: np.ndarray,
    snow_fraction: np.ndarray,
    snow_brightness: np.ndarray,
    snow_texture_strength: np.ndarray,
    elev_mod: np.ndarray,
    profile: dict[str, Any],
) -> None:
    settle_cfg = profile.get("settlement", profile["open_land"])
    lo, hi = settle_cfg["snow_fraction"]
    snow_fraction[mask] = np.clip(
        lo + (hi - lo) * 0.5 + elev_mod[mask] * 0.4,
        lo,
        hi,
    )
    snow_brightness[mask] = settle_cfg.get("snow_brightness", 0.88)
    snow_texture_strength[mask] = settle_cfg.get("snow_texture_strength", 0.25)


def _apply_forest(
    mask: np.ndarray,
    snow_fraction: np.ndarray,
    snow_texture_strength: np.ndarray,
    forest_snow_intensity: np.ndarray,
    elev_mod: np.ndarray,
    hillshade_mod: np.ndarray,
    profile: dict[str, Any],
    *,
    thickness_fraction: np.ndarray | None = None,
) -> None:
    forest_cfg = profile["forest"]
    flo, fhi = forest_cfg["snow_fraction"]
    if thickness_fraction is not None:
        canopy_factor = float(forest_cfg.get("canopy_thickness_factor", 0.82))
        snow_fraction[mask] = np.clip(
            flo + (fhi - flo) * thickness_fraction[mask] * canopy_factor,
            flo,
            fhi,
        )
    else:
        forest_base = (flo + fhi) / 2.0 + elev_mod[mask] * 0.42
        snow_fraction[mask] = np.clip(
            forest_base + hillshade_mod[mask] * 0.18,
            flo,
            fhi,
        )
    forest_snow_intensity[mask] = forest_cfg["forest_snow_intensity"]
    snow_texture_strength[mask] = forest_cfg.get("snow_texture_strength", 0.25)


def _apply_rock(
    mask: np.ndarray,
    snow_fraction: np.ndarray,
    rock_visibility: np.ndarray,
    slope: np.ndarray,
    rough: np.ndarray,
    tpi_mod: np.ndarray,
    aspect_mod: np.ndarray,
    profile: dict[str, Any],
    *,
    thickness_fraction: np.ndarray | None = None,
    burial_fraction: np.ndarray | None = None,
) -> None:
    rock_cfg = profile["rock"]
    gentle_max = float(rock_cfg.get("gentle_slope_max_deg", 28))
    steep_min = float(rock_cfg.get("steep_slope_min_deg", 42))
    slope_t = np.clip((slope - gentle_max) / max(steep_min - gentle_max, 1e-3), 0, 1)
    gentle_factor = 1.0 - slope_t

    rock_vis = np.clip(
        (slope - rock_cfg["slope_visibility_threshold_deg"]) / 30.0
        + (rough - rock_cfg["roughness_visibility_threshold"]) * 0.5,
        0,
        1,
    )
    min_vis = rock_cfg["min_rock_visibility"] * slope_t + 0.05 * gentle_factor
    rock_vis = np.maximum(rock_vis, min_vis)
    rock_vis = rock_vis * (0.12 + 0.88 * slope_t)

    snow_cover = burial_fraction if burial_fraction is not None else thickness_fraction
    if snow_cover is not None:
        burial_strength = float(rock_cfg.get("thickness_burial_factor", 0.82))
        burial = np.clip(snow_cover * burial_strength * gentle_factor, 0.0, 1.0)
        rock_vis = rock_vis * (1.0 - burial)
        rock_vis = np.maximum(rock_vis, min_vis * (1.0 - 0.75 * burial))

    rock_visibility[mask] = rock_vis[mask]

    gentle_boost = float(rock_cfg.get("gentle_snow_boost", 0.22))
    if thickness_fraction is not None:
        cover = snow_cover if snow_cover is not None else thickness_fraction
        rock_base = cover * rock_cfg["max_snow_fraction"]
        rock_snow = np.clip(
            rock_base
            - rock_vis * 0.20
            - aspect_mod * rock_cfg.get("aspect_south_penalty", 0.12)
            + gentle_factor * gentle_boost * 0.5,
            0.20,
            0.97,
        )
    else:
        rock_snow = np.clip(
            rock_cfg["max_snow_fraction"]
            - rock_vis * 0.20
            + np.clip(tpi_mod, 0, 1) * 0.18
            - aspect_mod * rock_cfg.get("aspect_south_penalty", 0.12)
            + gentle_factor * gentle_boost,
            0.20,
            0.97,
        )
    snow_fraction[mask] = rock_snow[mask]


def _apply_open_land(
    mask: np.ndarray,
    snow_fraction: np.ndarray,
    snow_brightness: np.ndarray,
    snow_texture_strength: np.ndarray,
    summer_exposure: np.ndarray,
    elev_mod: np.ndarray,
    profile: dict[str, Any],
    *,
    thickness_fraction: np.ndarray | None = None,
    slope_snow_scale: np.ndarray | None = None,
    protrusion_fraction: np.ndarray | None = None,
) -> None:
    open_cfg = profile["open_land"]
    lo, hi = open_cfg["snow_fraction"]
    if thickness_fraction is not None:
        cover = lo + (hi - lo) * thickness_fraction[mask]
    else:
        cover = lo + (hi - lo) * 0.68 + elev_mod[mask] * 0.5

    if slope_snow_scale is not None:
        cover = cover * slope_snow_scale[mask]

    if protrusion_fraction is not None:
        prot_red = float(open_cfg.get("protrusion_snow_reduction", 0.9))
        cover = cover * (1.0 - protrusion_fraction[mask] * prot_red)

    snow_fraction[mask] = np.clip(cover, 0.0, hi)
    snow_brightness[mask] = open_cfg["snow_brightness"]
    snow_texture_strength[mask] = open_cfg["snow_texture_strength"]

    exposure_vals = np.zeros(int(mask.sum()), dtype=np.float32)
    if slope_snow_scale is not None:
        steep_vis = float(open_cfg.get("slope_texture_visibility", 0.7))
        exposure_vals = np.maximum(exposure_vals, (1.0 - slope_snow_scale[mask]) * steep_vis)
    if protrusion_fraction is not None:
        prot_vis = float(open_cfg.get("protrusion_texture_visibility", 0.85))
        exposure_vals = np.maximum(exposure_vals, protrusion_fraction[mask] * prot_vis)
    summer_exposure[mask] = np.maximum(summer_exposure[mask], exposure_vals)


def _open_land_slope_snow_scale(slope: np.ndarray, open_cfg: dict[str, Any]) -> np.ndarray:
    """Reduce open-land snow on steep slopes; can go well below snow_fraction min."""
    start_deg = float(open_cfg.get("slope_snow_start_deg", 28.0))
    end_deg = float(open_cfg.get("slope_snow_end_deg", 40.0))
    min_scale = float(open_cfg.get("slope_min_snow_scale", 0.05))
    if end_deg <= start_deg:
        end_deg = start_deg + 1e-3
    t = np.clip((slope.astype(np.float64) - start_deg) / (end_deg - start_deg), 0.0, 1.0)
    return (1.0 - t * (1.0 - min_scale)).astype(np.float32)


def _terrain_protrusion_fraction(
    dem: np.ndarray,
    snow_surface_dem: np.ndarray,
    *,
    full_m: float,
) -> np.ndarray:
    """Where summer DEM rises above the snow deck (Geröll, Felsinseln, Mikrorelief)."""
    if full_m <= 0.0:
        return np.zeros(dem.shape, dtype=np.float32)
    protrusion = np.maximum(
        dem.astype(np.float32) - snow_surface_dem.astype(np.float32),
        0.0,
    )
    return np.clip(protrusion / full_m, 0.0, 1.0).astype(np.float32)


def _local_burial_fraction(
    snow_thickness: np.ndarray,
    base_height: float,
    *,
    resolution_m: float,
    radius_m: float,
) -> np.ndarray:
    """Neighborhood snow depth for burying small protrusions (stones, micro-bumps)."""
    radius_px = max(1, int(round(radius_m / resolution_m)))
    size = radius_px * 2 + 1
    local_thickness = ndimage.maximum_filter(snow_thickness, size=size)
    return np.clip(local_thickness / base_height, 0.0, 1.0).astype(np.float32)


def _normalize(array: np.ndarray) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - lo) / (hi - lo), 0, 1).astype(np.float32)


def _elevation_modifier(elevation: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    ref = float(cfg.get("reference_m", 1500))
    per_100m = float(cfg.get("snow_increase_per_100m", 0.03))
    max_boost = float(cfg.get("max_boost", 0.15))
    delta = (elevation - ref) / 100.0 * per_100m
    return np.clip(delta, -max_boost, max_boost).astype(np.float32)


def _aspect_modifier(aspect: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    south = float(cfg.get("south_thinning", 0.12))
    north = float(cfg.get("north_boost", 0.08))
    south_factor = np.cos(np.radians(aspect - 180)) * south
    north_factor = np.cos(np.radians(aspect)) * north
    return (north_factor - south_factor).astype(np.float32)
