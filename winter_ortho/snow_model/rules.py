from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy import ndimage

from winter_ortho.preprocessing.tiling import get_tile_grid
from winter_ortho.snow_model.surface import compute_snow_cover_weight, resolve_snow_surface_config
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
    blanket_thickness: np.ndarray | None = None,
    accumulation_mask: np.ndarray | None = None,
    snow_surface_dem: np.ndarray | None = None,
    snow_cover_weight: np.ndarray | None = None,
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
    rock_cover_fraction = None
    slope_snow_scale = None
    protrusion_fraction = None
    if use_thickness:
        surface_cfg = resolve_snow_surface_config(profile)
        base_height = max(surface_cfg["base_snow_height_m"], 1e-3)
        effective_depth = _effective_snow_depth(
            snow_thickness,
            blanket_thickness,
            accumulation_mask,
        )
        assert effective_depth is not None
        thickness_fraction = np.clip(effective_depth / base_height, 0.0, 1.0).astype(np.float32)
        resolution_m = float(config.get("resolution_m", 2.0))
        rock_cfg = profile.get("rock", {})
        burial_radius_m = float(rock_cfg.get("thickness_burial_radius_m", 20.0))
        burial_fraction = _local_burial_fraction(
            effective_depth,
            base_height,
            resolution_m=resolution_m,
            radius_m=burial_radius_m,
        )
        if blanket_thickness is not None:
            full_m = max(
                float(
                    rock_cfg.get(
                        "full_snow_thickness_m",
                        profile.get("open_land", {}).get("full_snow_thickness_m", 0.5),
                    )
                ),
                1e-3,
            )
            rock_cover_fraction = np.clip(blanket_thickness / full_m, 0.0, 1.0).astype(np.float32)

    open_cfg = profile.get("open_land", {})
    slope = terrain["slope"]
    resolution_m = float(config.get("resolution_m", 2.0))
    if use_thickness and snow_cover_weight is None:
        snow_cover_weight = compute_snow_cover_weight(
            slope,
            resolve_snow_surface_config(profile),
            resolution_m=resolution_m,
        )
    if use_thickness and snow_surface_dem is not None:
        if float(open_cfg.get("slope_snow_strength", 1.0)) > 0.0:
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
        rock_cover_fraction=rock_cover_fraction,
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
        snow_thickness_m=snow_thickness,
        blanket_thickness_m=blanket_thickness,
        accumulation_mask=accumulation_mask,
        snow_cover_weight=snow_cover_weight,
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
    rock_cover_fraction: np.ndarray | None = None,
) -> None:
    rock_cfg = profile["rock"]
    gentle_max = float(rock_cfg.get("gentle_slope_max_deg", 28))
    steep_min = float(rock_cfg.get("steep_slope_min_deg", 42))
    slope_t = np.clip((slope - gentle_max) / max(steep_min - gentle_max, 1e-3), 0, 1)
    gentle_factor = 1.0 - slope_t
    max_snow = float(rock_cfg["max_snow_fraction"])

    rock_vis = np.clip(
        (slope - rock_cfg["slope_visibility_threshold_deg"]) / 30.0
        + (rough - rock_cfg["roughness_visibility_threshold"]) * 0.5,
        0,
        1,
    )
    min_vis = rock_cfg["min_rock_visibility"] * slope_t + 0.05 * gentle_factor
    rock_vis = np.maximum(rock_vis, min_vis)
    rock_vis = rock_vis * (0.12 + 0.88 * slope_t)

    snow_cover = rock_cover_fraction
    if snow_cover is None:
        snow_cover = burial_fraction if burial_fraction is not None else thickness_fraction
    thick_enough = rock_cover_fraction >= 1.0 if rock_cover_fraction is not None else None
    if snow_cover is not None:
        burial_strength = float(rock_cfg.get("thickness_burial_factor", 0.82))
        burial = np.clip(snow_cover * burial_strength * gentle_factor, 0.0, 1.0)
        rock_vis = rock_vis * (1.0 - burial)
        rock_vis = np.maximum(rock_vis, min_vis * (1.0 - 0.75 * burial))
        if thick_enough is not None:
            rock_vis = np.where(
                thick_enough,
                rock_vis * (0.22 + 0.78 * slope_t),
                rock_vis,
            )

    rock_visibility[mask] = rock_vis[mask]

    gentle_boost = float(rock_cfg.get("gentle_snow_boost", 0.22))
    vis_penalty = 0.20 if thick_enough is None else np.where(thick_enough, 0.05, 0.20)

    # Slope-dependent cap: even with full snow cover, steep rock stays partially
    # visible. gentle_max → no reduction; steep_min → reduced by steep_snow_cap.
    steep_snow_cap = float(rock_cfg.get("steep_snow_cap", 0.0))
    if steep_snow_cap > 0.0:
        slope_cap = max_snow * (1.0 - slope_t * steep_snow_cap)
    else:
        slope_cap = np.full_like(slope_t, max_snow)

    if thickness_fraction is not None:
        cover = snow_cover if snow_cover is not None else thickness_fraction
        if rock_cover_fraction is not None:
            rock_base = np.where(thick_enough, slope_cap, cover * slope_cap)
        else:
            rock_base = cover * slope_cap
        rock_snow = np.clip(
            rock_base
            - rock_vis * vis_penalty
            - aspect_mod * rock_cfg.get("aspect_south_penalty", 0.12)
            + gentle_factor * gentle_boost * 0.5,
            0.20,
            slope_cap,
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
    snow_thickness_m: np.ndarray | None = None,
    blanket_thickness_m: np.ndarray | None = None,
    accumulation_mask: np.ndarray | None = None,
    snow_cover_weight: np.ndarray | None = None,
    slope_snow_scale: np.ndarray | None = None,
    protrusion_fraction: np.ndarray | None = None,
) -> None:
    open_cfg = profile["open_land"]
    lo, hi = open_cfg["snow_fraction"]
    full_m = max(float(open_cfg.get("full_snow_thickness_m", 0.5)), 1e-3)
    depth_source = _open_land_depth_for_gating(
        mask,
        snow_thickness_m=snow_thickness_m,
        blanket_thickness_m=blanket_thickness_m,
        accumulation_mask=accumulation_mask,
        snow_cover_weight=snow_cover_weight,
        deck_depth_cover_floor=float(open_cfg.get("deck_depth_cover_floor", 0.0)),
    )
    if depth_source is not None:
        depth = depth_source
        depth_frac = np.clip(depth / full_m, 0.0, 1.0)
        thick_enough = depth >= full_m
        cover = np.where(thick_enough, hi, lo + (hi - lo) * depth_frac).astype(np.float32)
        thin_gate = np.where(thick_enough, 0.0, 1.0 - depth_frac).astype(np.float32)
    elif thickness_fraction is not None:
        cover = lo + (hi - lo) * thickness_fraction[mask]
        thin_gate = 1.0 - thickness_fraction[mask]
    else:
        cover = lo + (hi - lo) * 0.68 + elev_mod[mask] * 0.5
        thin_gate = np.ones(int(mask.sum()), dtype=np.float32)

    if slope_snow_scale is not None:
        slope_strength = float(open_cfg.get("slope_snow_strength", 1.0))
        steep_factor = (1.0 - slope_snow_scale[mask]).astype(np.float32)
        min_steep = open_cfg.get("slope_min_snow_fraction")
        if min_steep is not None:
            penalty = steep_factor * slope_strength
            reduced = cover * (1.0 - penalty)
            floor_band = float(open_cfg.get("slope_min_snow_softness", 0.04))
            effective_min = np.full(int(mask.sum()), float(min_steep), dtype=np.float32)
            if snow_cover_weight is not None:
                deck = np.clip(snow_cover_weight[mask], 0.0, 1.0)
                boost = float(open_cfg.get("deck_snow_fraction_boost", 0.75))
                effective_min = np.minimum(
                    float(min_steep)
                    + (hi - float(min_steep)) * deck * boost,
                    hi,
                ).astype(np.float32)
            cover = _soft_floor(reduced, effective_min, floor_band)
        else:
            penalty = steep_factor * thin_gate * slope_strength
            cover = cover * (1.0 - penalty)

    if protrusion_fraction is not None and open_cfg.get("slope_min_snow_fraction") is None:
        prot_red = float(open_cfg.get("protrusion_snow_reduction", 0.9))
        prot_strength = float(open_cfg.get("protrusion_strength", 1.0))
        prot_frac = protrusion_fraction[mask]
        if snow_cover_weight is not None:
            deck = np.clip(snow_cover_weight[mask], 0.0, 1.0)
            prot_frac = prot_frac * np.clip((deck - 0.12) / 0.88, 0.0, 1.0)
        cover = cover * (
            1.0 - prot_frac * prot_red * prot_strength * thin_gate
        )

    snow_fraction[mask] = np.clip(cover, 0.0, hi)
    snow_brightness[mask] = open_cfg["snow_brightness"]
    snow_texture_strength[mask] = open_cfg["snow_texture_strength"]

    exposure_vals = np.zeros(int(mask.sum()), dtype=np.float32)
    if slope_snow_scale is not None and open_cfg.get("slope_min_snow_fraction") is None:
        steep_vis = float(open_cfg.get("slope_texture_visibility", 0.7))
        slope_strength = float(open_cfg.get("slope_snow_strength", 1.0))
        steep_factor = (1.0 - slope_snow_scale[mask]).astype(np.float32)
        exposure_vals = np.maximum(
            exposure_vals,
            steep_factor * steep_vis * thin_gate * slope_strength,
        )
    if protrusion_fraction is not None and open_cfg.get("slope_min_snow_fraction") is None:
        prot_vis = float(open_cfg.get("protrusion_texture_visibility", 0.85))
        prot_strength = float(open_cfg.get("protrusion_strength", 1.0))
        exposure_vals = np.maximum(
            exposure_vals,
            protrusion_fraction[mask] * prot_vis * prot_strength * thin_gate,
        )
    summer_exposure[mask] = np.maximum(summer_exposure[mask], exposure_vals)


def _effective_snow_depth(
    snow_thickness: np.ndarray | None,
    blanket_thickness: np.ndarray | None,
    accumulation_mask: np.ndarray | None,
) -> np.ndarray | None:
    """Nominal blanket depth in accumulation zones; geometric depth on steep faces."""
    if snow_thickness is None:
        return None
    if blanket_thickness is not None and accumulation_mask is not None:
        on_accum = accumulation_mask > 0
        return np.where(on_accum, blanket_thickness, snow_thickness).astype(np.float32)
    if blanket_thickness is not None:
        return blanket_thickness.astype(np.float32)
    return snow_thickness.astype(np.float32)


def _soft_floor(values: np.ndarray, floor: np.ndarray | float, band: float) -> np.ndarray:
    """Ease values up to a floor instead of a hard clip."""
    floor_arr = (
        np.broadcast_to(floor, values.shape).astype(np.float64)
        if np.ndim(floor) != 0
        else float(floor)
    )
    delta = floor_arr - values.astype(np.float64)
    t = np.clip(delta / max(band, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return (values * (1.0 - smooth) + floor_arr * smooth).astype(np.float32)


def _open_land_depth_for_gating(
    mask: np.ndarray,
    *,
    snow_thickness_m: np.ndarray | None,
    blanket_thickness_m: np.ndarray | None,
    accumulation_mask: np.ndarray | None,
    snow_cover_weight: np.ndarray | None = None,
    deck_depth_cover_floor: float = 0.0,
) -> np.ndarray | None:
    if snow_thickness_m is not None:
        depth = snow_thickness_m.astype(np.float32)
    else:
        depth = _effective_snow_depth(snow_thickness_m, blanket_thickness_m, accumulation_mask)
    if depth is None:
        return None
    if blanket_thickness_m is not None and snow_cover_weight is not None:
        deck_cover = snow_cover_weight.astype(np.float32)
        if deck_depth_cover_floor > 0.0:
            deck_cover = np.where(
                deck_cover > 0.05,
                np.maximum(deck_cover, deck_depth_cover_floor),
                deck_cover,
            )
        deck_depth = (blanket_thickness_m * deck_cover).astype(np.float32)
        if snow_thickness_m is not None:
            deck_depth = np.where(
                snow_thickness_m > 0.05,
                np.minimum(deck_depth, snow_thickness_m),
                deck_depth,
            ).astype(np.float32)
        depth = np.where(
            deck_cover > 0.05,
            np.maximum(depth, deck_depth),
            depth,
        ).astype(np.float32)
    elif (
        blanket_thickness_m is not None
        and accumulation_mask is not None
        and snow_thickness_m is not None
    ):
        on_accum = accumulation_mask > 0
        depth = np.where(
            on_accum,
            np.maximum(depth, blanket_thickness_m),
            depth,
        ).astype(np.float32)
    return depth[mask]


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
