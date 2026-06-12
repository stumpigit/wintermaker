from __future__ import annotations

from typing import Any

import numpy as np

from winter_ortho.features.terrain import compute_generalized_hillshade, hillshade_config_for_render
from winter_ortho.snow_model.surface import resolve_snow_surface_config
from winter_ortho.preprocessing.tiling import get_tile_grid
from winter_ortho.rendering.base import to_float_rgb, to_uint8_rgb
from winter_ortho.rendering.buildings import render_buildings
from winter_ortho.rendering.forest import render_forest
from winter_ortho.rendering.open_land import render_open_land
from winter_ortho.rendering.rock import render_rock
from winter_ortho.rendering.paths import render_paths
from winter_ortho.rendering.roads import render_roads
from winter_ortho.rendering.settlement import render_settlement
from winter_ortho.rendering.relief import apply_map_shading, apply_winter_relief
from winter_ortho.rendering.summer_structure import (
    apply_summer_anchored_grade,
    apply_summer_cast_shadows,
    summer_cast_shadow_field,
)
from winter_ortho.rendering.water import render_water
from winter_ortho.utils.config import load_class_rules
from winter_ortho.utils.paths import TilePaths
from winter_ortho.utils.progress import PipelineProgress
from winter_ortho.utils.raster import read_raster, write_cog

RENDER_STEPS = [
    "water",
    "open_land",
    "settlement",
    "rock",
    "forest",
    "roads",
    "paths",
    "buildings",
]


def blend_hillshade_for_snow(
    hillshade_base: np.ndarray,
    hillshade_snow: np.ndarray,
    snow_fraction: np.ndarray,
    slope: np.ndarray,
    *,
    max_accumulation_slope_deg: float,
    accumulation_mask: np.ndarray | None = None,
) -> np.ndarray:
    if accumulation_mask is None:
        accumulation = slope < max_accumulation_slope_deg
    else:
        accumulation = accumulation_mask > 0
    weight = np.clip(snow_fraction, 0.0, 1.0) * accumulation.astype(np.float32)
    return (
        hillshade_base * (1.0 - weight) + hillshade_snow * weight
    ).astype(np.float32)


def render_winter_tile(
    config: dict[str, Any],
    profile: dict[str, Any],
    paths: TilePaths,
    class_masks: dict[str, np.ndarray],
    snow_layers: dict[str, np.ndarray],
    terrain: dict[str, np.ndarray],
    *,
    snow_surface: dict[str, np.ndarray] | None = None,
    progress: PipelineProgress | None = None,
) -> np.ndarray:
    grid = get_tile_grid(config, paths.tile_id)
    rgb_raw, _ = read_raster(str(paths.rgb_summer))
    summer_rgb = to_float_rgb(rgb_raw)
    rgb = summer_rgb.copy()

    landcover, _ = read_raster(str(paths.landcover_mask))
    landcover = landcover[0]
    mask_values = load_class_rules()["mask_values"]
    exclusive_masks = {
        name: (landcover == value).astype(np.uint8)
        for name, value in mask_values.items()
    }

    render_cfg = profile.get("rendering", {})
    snow_color = np.array(render_cfg.get("snow_color", [0.94, 0.96, 0.98]), dtype=np.float32)
    road_color = np.array(render_cfg.get("road_color", [0.78, 0.80, 0.82]), dtype=np.float32)
    summer_cfg = render_cfg.get("summer_structure", {})
    cast_cfg = render_cfg.get("cast_shadows", {})
    map_shade_cfg = render_cfg.get("map_shading", {})
    terrain_cfg = config.get("terrain", {})
    render_hillshade_cfg = hillshade_config_for_render(terrain_cfg, map_shade_cfg)
    resolution_m = float(config.get("resolution_m", 2.0))
    hill_generalized = compute_generalized_hillshade(
        terrain["elevation"],
        resolution_m,
        render_hillshade_cfg,
    )
    sun_azimuth = float(render_hillshade_cfg["hillshade"]["azimuth"])
    render_hillshade = hill_generalized
    if snow_surface is not None and "snow_surface" in profile:
        surface_cfg = resolve_snow_surface_config(profile)
        hill_snow = compute_generalized_hillshade(
            snow_surface["snow_surface_dem"],
            resolution_m,
            render_hillshade_cfg,
        )
        render_hillshade = blend_hillshade_for_snow(
            hill_generalized,
            hill_snow,
            snow_layers["snow_fraction"],
            terrain["slope"],
            max_accumulation_slope_deg=surface_cfg["max_accumulation_slope_deg"],
            accumulation_mask=snow_surface.get("accumulation_mask"),
        )
    cast_shadow = summer_cast_shadow_field(
        summer_rgb,
        building_mask=exclusive_masks.get("building_mask"),
        forest_mask=exclusive_masks.get("forest_mask"),
        reference_sigma_px=float(cast_cfg.get("reference_sigma_px", 32.0)),
        fine_sigma_px=float(cast_cfg.get("fine_sigma_px", 6.0)),
        building_radius_px=int(cast_cfg.get("building_radius_px", 10)),
        building_boost=float(cast_cfg.get("building_boost", 1.6)),
        forest_boost=float(cast_cfg.get("forest_boost", 1.25)),
        base_strength=float(cast_cfg.get("base_strength", 0.35)),
    )

    if progress:
        progress.substep("Rendering class layers: " + " → ".join(RENDER_STEPS))

    result = render_water(
        rgb,
        exclusive_masks["water_mask"],
        darken=float(profile["water"].get("water_darken", 0.85)),
    )
    open_cfg = profile.get("open_land", {})
    result = render_open_land(
        result,
        exclusive_masks["open_land_mask"],
        snow_fraction=snow_layers["snow_fraction"],
        snow_brightness=snow_layers["snow_brightness"],
        snow_texture_strength=snow_layers["snow_texture_strength"],
        summer_exposure=snow_layers.get("summer_exposure"),
        hillshade=render_hillshade,
        hillshade_generalized=hill_generalized,
        cast_shadow=cast_shadow,
        snow_color=snow_color,
        hillshade_strength=float(render_cfg.get("hillshade_strength_open_land", 0.12)),
        hillshade_compression=float(open_cfg.get("hillshade_compression", 0.28)),
        snow_flattening=float(open_cfg.get("snow_flattening", 0.82)),
        original_texture_visibility=float(open_cfg.get("original_texture_visibility", 0.30)),
        max_snow_blend=float(open_cfg.get("max_snow_blend", 0.72)),
        detail_preservation=float(open_cfg.get("detail_preservation", 0.55)),
        noise_scale_px=int(render_cfg.get("noise_scale_px", 8)),
        summer_shade_weight=float(open_cfg.get("summer_shade_weight", 0.62)),
        hillshade_shade_weight=float(open_cfg.get("hillshade_shade_weight", 0.38)),
        macro_hillshade_weight=float(open_cfg.get("macro_hillshade_weight", 0.88)),
        cast_shadow_weight=float(open_cfg.get("cast_shadow_weight", 0.28)),
        shadow_boost=float(open_cfg.get("shadow_boost", 1.35)),
        highlight_cap=float(open_cfg.get("highlight_cap", 0.45)),
    )
    settle_cfg = profile.get("settlement", {})
    result = render_settlement(
        result,
        exclusive_masks["settlement_mask"],
        snow_fraction=snow_layers["snow_fraction"],
        snow_brightness=snow_layers["snow_brightness"],
        snow_texture_strength=snow_layers["snow_texture_strength"],
        snow_color=snow_color,
        hillshade=render_hillshade,
        hillshade_generalized=hill_generalized,
        cast_shadow=cast_shadow,
        hillshade_strength=float(render_cfg.get("hillshade_strength_settlement", 0.14)),
        hillshade_compression=float(settle_cfg.get("hillshade_compression", 0.35)),
        snow_flattening=float(settle_cfg.get("snow_flattening", 0.60)),
        original_texture_visibility=float(settle_cfg.get("original_texture_visibility", 0.25)),
        max_snow_blend=float(settle_cfg.get("max_snow_blend", 0.70)),
        noise_scale_px=int(render_cfg.get("noise_scale_px", 8)),
        summer_shade_weight=float(settle_cfg.get("summer_shade_weight", 0.58)),
        hillshade_shade_weight=float(settle_cfg.get("hillshade_shade_weight", 0.42)),
        macro_hillshade_weight=float(settle_cfg.get("macro_hillshade_weight", 0.85)),
        cast_shadow_weight=float(settle_cfg.get("cast_shadow_weight", 0.32)),
        shadow_boost=float(settle_cfg.get("shadow_boost", 1.25)),
        highlight_cap=float(settle_cfg.get("highlight_cap", 0.48)),
    )
    rock_cfg = profile.get("rock", {})
    result = render_rock(
        result,
        exclusive_masks["rock_or_bare_ground_mask"],
        snow_fraction=snow_layers["snow_fraction"],
        rock_visibility=snow_layers["rock_visibility"],
        slope=terrain["slope"],
        snow_color=snow_color,
        hillshade=render_hillshade,
        hillshade_strength=float(render_cfg.get("hillshade_strength_rock", 0.18)),
        hillshade_compression=float(rock_cfg.get("hillshade_compression", 0.50)),
        snow_flattening=float(rock_cfg.get("snow_flattening", 0.40)),
        gentle_slope_max_deg=float(rock_cfg.get("gentle_slope_max_deg", 28)),
        steep_slope_min_deg=float(rock_cfg.get("steep_slope_min_deg", 42)),
        gentle_snow_boost=float(rock_cfg.get("gentle_render_boost", 0.18)),
        summer_shade_weight=float(rock_cfg.get("summer_shade_weight", 0.55)),
        hillshade_shade_weight=float(rock_cfg.get("hillshade_shade_weight", 0.45)),
        shadow_boost=float(rock_cfg.get("shadow_boost", 1.45)),
        highlight_cap=float(rock_cfg.get("highlight_cap", 0.50)),
        summer_preservation=float(rock_cfg.get("summer_preservation", 0.42)),
    )
    forest_cfg = profile.get("forest", {})
    result = render_forest(
        result,
        exclusive_masks["forest_mask"],
        snow_fraction=snow_layers["snow_fraction"],
        forest_snow_intensity=snow_layers["forest_snow_intensity"],
        snow_color=snow_color,
        hillshade=render_hillshade,
        contrast_reduction=float(forest_cfg.get("contrast_reduction", 0.10)),
        original_texture_visibility=float(forest_cfg.get("original_texture_visibility", 0.08)),
        crown_tophat_radius_px=int(forest_cfg.get("crown_tophat_radius_px", 4)),
        hillshade_strength=float(render_cfg.get("hillshade_strength_forest", 0.18)),
        crown_highlight_strength=float(forest_cfg.get("crown_highlight_strength", 0.42)),
        crown_noise_strength=float(forest_cfg.get("crown_noise_strength", 0.04)),
        noise_scale_px=int(render_cfg.get("noise_scale_px", 8)),
        winter_luminance_lo=float(forest_cfg.get("winter_luminance_lo", 0.08)),
        winter_luminance_hi=float(forest_cfg.get("winter_luminance_hi", 0.38)),
        summer_structure_strength=float(forest_cfg.get("summer_structure_strength", 0.82)),
        green_suppression=float(forest_cfg.get("green_suppression", 0.78)),
        max_crown_snow_alpha=float(forest_cfg.get("max_crown_snow_alpha", 0.48)),
        canopy_blanket=float(forest_cfg.get("canopy_blanket", 0.50)),
    )
    roads_cfg = profile.get("roads", {})
    result = render_roads(
        result,
        exclusive_masks["road_mask"],
        summer_rgb,
        snow_fraction=snow_layers["snow_fraction"],
        road_visibility=snow_layers["road_visibility"],
        road_color=road_color,
        summer_line_strength=float(roads_cfg.get("summer_line_strength", 0.72)),
        min_visibility=float(roads_cfg.get("min_visibility", 0.38)),
    )
    paths_cfg = profile.get("paths", {})
    result = render_paths(
        result,
        exclusive_masks["path_mask"],
        snow_fraction=snow_layers["snow_fraction"],
        snow_brightness=snow_layers["snow_brightness"],
        snow_texture_strength=snow_layers["snow_texture_strength"],
        snow_color=snow_color,
        hillshade=render_hillshade,
        bury_strength=float(paths_cfg.get("bury_strength", 0.90)),
        max_snow_blend=float(paths_cfg.get("max_snow_blend", 0.82)),
        hillshade_strength=float(render_cfg.get("hillshade_strength_open_land", 0.12)),
        hillshade_compression=float(open_cfg.get("hillshade_compression", 0.28)),
        snow_flattening=float(open_cfg.get("snow_flattening", 0.82)),
        noise_scale_px=int(render_cfg.get("noise_scale_px", 8)),
    )
    buildings_cfg = profile.get("buildings", {})
    result = render_buildings(
        result,
        exclusive_masks["building_mask"],
        summer_rgb,
        roof_snow_intensity=snow_layers["roof_snow_intensity"],
        snow_color=snow_color,
        brighten_factor=float(buildings_cfg.get("brighten_factor", 0.55)),
        edge_preservation=float(buildings_cfg.get("edge_preservation", 0.95)),
        wall_preservation=float(buildings_cfg.get("wall_preservation", 0.62)),
    )

    relief_cfg = render_cfg.get("relief", {})
    class_weights = relief_cfg.get("class_weights", {})
    relief_weight = np.zeros(snow_layers["snow_fraction"].shape, dtype=np.float32)
    weight_map = [
        ("open_land_mask", class_weights.get("open_land", 0.14)),
        ("settlement_mask", class_weights.get("settlement", 0.22)),
        ("rock_or_bare_ground_mask", class_weights.get("rock", 0.62)),
        ("path_mask", class_weights.get("paths", 0.10)),
        ("building_mask", class_weights.get("buildings", 0.35)),
    ]
    for mask_name, weight in weight_map:
        class_mask = exclusive_masks.get(mask_name)
        if class_mask is not None:
            relief_weight[class_mask > 0] = weight

    result = apply_winter_relief(
        result,
        hillshade=render_hillshade,
        hillshade_generalized=hill_generalized,
        cast_shadow=cast_shadow,
        aspect=terrain["aspect"],
        snow_fraction=snow_layers["snow_fraction"],
        snow_color=snow_color,
        summer_rgb=summer_rgb,
        relief_weight=relief_weight,
        hillshade_strength=float(relief_cfg.get("hillshade_strength", 0.10)),
        aspect_strength=0.0,
        compression=float(relief_cfg.get("compression", 0.32)),
        macro_hillshade_weight=float(relief_cfg.get("macro_hillshade_weight", 0.85)),
        min_snow=float(relief_cfg.get("min_snow", 0.35)),
        shade_min=float(relief_cfg.get("shade_min", 0.88)),
        sun_max=float(relief_cfg.get("sun_max", 1.12)),
        snow_tint_strength=float(relief_cfg.get("snow_tint_strength", 0.50)),
        summer_relief_weight=float(relief_cfg.get("summer_relief_weight", 0.0)),
        cast_shadow_relief_weight=float(relief_cfg.get("cast_shadow_relief_weight", 0.55)),
        sun_azimuth=sun_azimuth,
        south_shadow_boost=float(map_shade_cfg.get("south_shadow_boost", 1.0)),
    )

    result = apply_summer_anchored_grade(
        result,
        summer_rgb,
        exclusive_masks,
        summer_cfg,
    )

    result = apply_summer_cast_shadows(
        result,
        cast_shadow,
        snow_fraction=snow_layers["snow_fraction"],
        strength=float(cast_cfg.get("final_strength", 0.62)),
        min_snow=float(cast_cfg.get("min_snow", 0.22)),
        max_darken=float(cast_cfg.get("max_darken", 0.48)),
    )

    protect_map = np.zeros(snow_layers["snow_fraction"].shape, dtype=np.uint8)
    for mask_name in map_shade_cfg.get("protect_masks", ["water_mask"]):
        class_mask = exclusive_masks.get(mask_name)
        if class_mask is not None:
            protect_map |= class_mask
    result = apply_map_shading(
        result,
        hillshade_generalized=hill_generalized,
        aspect=terrain["aspect"],
        slope=terrain["slope"],
        snow_fraction=snow_layers["snow_fraction"],
        sun_azimuth=sun_azimuth,
        strength=float(map_shade_cfg.get("strength", 0.50)),
        hillshade_weight=float(map_shade_cfg.get("hillshade_weight", 0.72)),
        aspect_weight=float(map_shade_cfg.get("aspect_weight", 0.28)),
        south_shadow_boost=float(map_shade_cfg.get("south_shadow_boost", 1.35)),
        compression=float(map_shade_cfg.get("compression", 0.88)),
        min_snow=float(map_shade_cfg.get("min_snow", 0.12)),
        shade_min=float(map_shade_cfg.get("shade_min", 0.55)),
        sun_max=float(map_shade_cfg.get("sun_max", 1.35)),
        protect_mask=protect_map,
    )

    winter_uint8 = np.moveaxis(to_uint8_rgb(result), -1, 0)
    if progress:
        progress.substep(f"Writing {paths.winter_rgb.name}")
    write_cog(
        str(paths.winter_rgb),
        winter_uint8,
        transform=grid.transform,
        crs=grid.crs,
        nodata=None,
    )
    return winter_uint8
