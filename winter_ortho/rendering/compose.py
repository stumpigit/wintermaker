from __future__ import annotations

from typing import Any

import numpy as np

from winter_ortho.preprocessing.tiling import get_tile_grid
from winter_ortho.rendering.base import to_float_rgb, to_uint8_rgb
from winter_ortho.rendering.buildings import render_buildings
from winter_ortho.rendering.forest import render_forest
from winter_ortho.rendering.open_land import render_open_land
from winter_ortho.rendering.rock import render_rock
from winter_ortho.rendering.paths import render_paths
from winter_ortho.rendering.roads import render_roads
from winter_ortho.rendering.settlement import render_settlement
from winter_ortho.rendering.relief import apply_winter_relief
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


def render_winter_tile(
    config: dict[str, Any],
    profile: dict[str, Any],
    paths: TilePaths,
    class_masks: dict[str, np.ndarray],
    snow_layers: dict[str, np.ndarray],
    terrain: dict[str, np.ndarray],
    *,
    progress: PipelineProgress | None = None,
) -> np.ndarray:
    grid = get_tile_grid(config, paths.tile_id)
    rgb_raw, _ = read_raster(str(paths.rgb_summer))
    rgb = to_float_rgb(rgb_raw)

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
        hillshade=terrain["hillshade_winter_low_sun"],
        snow_color=snow_color,
        hillshade_strength=float(render_cfg.get("hillshade_strength_open_land", 0.12)),
        hillshade_compression=float(open_cfg.get("hillshade_compression", 0.28)),
        snow_flattening=float(open_cfg.get("snow_flattening", 0.82)),
        original_texture_visibility=float(open_cfg.get("original_texture_visibility", 0.30)),
        max_snow_blend=float(open_cfg.get("max_snow_blend", 0.72)),
        detail_preservation=float(open_cfg.get("detail_preservation", 0.55)),
        noise_scale_px=int(render_cfg.get("noise_scale_px", 8)),
    )
    settle_cfg = profile.get("settlement", {})
    result = render_settlement(
        result,
        exclusive_masks["settlement_mask"],
        snow_fraction=snow_layers["snow_fraction"],
        snow_brightness=snow_layers["snow_brightness"],
        snow_texture_strength=snow_layers["snow_texture_strength"],
        snow_color=snow_color,
        hillshade=terrain["hillshade_winter_low_sun"],
        hillshade_strength=float(render_cfg.get("hillshade_strength_settlement", 0.14)),
        hillshade_compression=float(settle_cfg.get("hillshade_compression", 0.35)),
        snow_flattening=float(settle_cfg.get("snow_flattening", 0.60)),
        original_texture_visibility=float(settle_cfg.get("original_texture_visibility", 0.25)),
        max_snow_blend=float(settle_cfg.get("max_snow_blend", 0.70)),
        noise_scale_px=int(render_cfg.get("noise_scale_px", 8)),
    )
    rock_cfg = profile.get("rock", {})
    result = render_rock(
        result,
        exclusive_masks["rock_or_bare_ground_mask"],
        snow_fraction=snow_layers["snow_fraction"],
        rock_visibility=snow_layers["rock_visibility"],
        slope=terrain["slope"],
        snow_color=snow_color,
        hillshade=terrain["hillshade_winter_low_sun"],
        hillshade_strength=float(render_cfg.get("hillshade_strength_rock", 0.18)),
        hillshade_compression=float(rock_cfg.get("hillshade_compression", 0.50)),
        snow_flattening=float(rock_cfg.get("snow_flattening", 0.40)),
        gentle_slope_max_deg=float(rock_cfg.get("gentle_slope_max_deg", 28)),
        steep_slope_min_deg=float(rock_cfg.get("steep_slope_min_deg", 42)),
        gentle_snow_boost=float(rock_cfg.get("gentle_render_boost", 0.18)),
    )
    forest_cfg = profile.get("forest", {})
    result = render_forest(
        result,
        exclusive_masks["forest_mask"],
        snow_fraction=snow_layers["snow_fraction"],
        forest_snow_intensity=snow_layers["forest_snow_intensity"],
        snow_color=snow_color,
        hillshade=terrain["hillshade_winter_low_sun"],
        contrast_reduction=float(forest_cfg.get("contrast_reduction", 0.10)),
        original_texture_visibility=float(forest_cfg.get("original_texture_visibility", 0.08)),
        crown_tophat_radius_px=int(forest_cfg.get("crown_tophat_radius_px", 4)),
        hillshade_strength=float(render_cfg.get("hillshade_strength_forest", 0.18)),
        crown_highlight_strength=float(forest_cfg.get("crown_highlight_strength", 0.42)),
        crown_noise_strength=float(forest_cfg.get("crown_noise_strength", 0.04)),
        noise_scale_px=int(render_cfg.get("noise_scale_px", 8)),
    )
    result = render_roads(
        result,
        exclusive_masks["road_mask"],
        snow_fraction=snow_layers["snow_fraction"],
        road_visibility=snow_layers["road_visibility"],
        road_color=road_color,
    )
    paths_cfg = profile.get("paths", {})
    result = render_paths(
        result,
        exclusive_masks["path_mask"],
        snow_fraction=snow_layers["snow_fraction"],
        snow_brightness=snow_layers["snow_brightness"],
        snow_texture_strength=snow_layers["snow_texture_strength"],
        snow_color=snow_color,
        hillshade=terrain["hillshade_winter_low_sun"],
        bury_strength=float(paths_cfg.get("bury_strength", 0.90)),
        max_snow_blend=float(paths_cfg.get("max_snow_blend", 0.82)),
        hillshade_strength=float(render_cfg.get("hillshade_strength_open_land", 0.12)),
        hillshade_compression=float(open_cfg.get("hillshade_compression", 0.28)),
        snow_flattening=float(open_cfg.get("snow_flattening", 0.82)),
        noise_scale_px=int(render_cfg.get("noise_scale_px", 8)),
    )
    result = render_buildings(
        result,
        exclusive_masks["building_mask"],
        roof_snow_intensity=snow_layers["roof_snow_intensity"],
        snow_color=snow_color,
        brighten_factor=float(profile["buildings"].get("brighten_factor", 0.55)),
        edge_preservation=float(profile["buildings"].get("edge_preservation", 0.95)),
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
        hillshade=terrain["hillshade_winter_low_sun"],
        aspect=terrain["aspect"],
        snow_fraction=snow_layers["snow_fraction"],
        snow_color=snow_color,
        relief_weight=relief_weight,
        hillshade_strength=float(relief_cfg.get("hillshade_strength", 0.10)),
        aspect_strength=float(relief_cfg.get("aspect_strength", 0.06)),
        compression=float(relief_cfg.get("compression", 0.32)),
        min_snow=float(relief_cfg.get("min_snow", 0.35)),
        shade_min=float(relief_cfg.get("shade_min", 0.88)),
        sun_max=float(relief_cfg.get("sun_max", 1.12)),
        snow_tint_strength=float(relief_cfg.get("snow_tint_strength", 0.50)),
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
