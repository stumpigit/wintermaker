from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from winter_ortho.io.tlm3d import load_tlm_layer
from winter_ortho.preprocessing.rasterize_vectors import apply_priority, rasterize_layer
from winter_ortho.preprocessing.tiling import get_tile_bbox, get_tile_grid
from winter_ortho.utils.paths import TilePaths
from winter_ortho.utils.progress import PipelineProgress
from winter_ortho.utils.raster import TargetGrid, read_raster, write_cog


def _road_buffer(layer_cfg: dict[str, Any], row: Any) -> float:
    category_field = layer_cfg.get("category_field")
    categories = layer_cfg.get("categories", {})
    default_width = float(layer_cfg.get("default_width_m", 6.0))
    if category_field and category_field in row.index:
        category = str(row[category_field])
        return float(categories.get(category, default_width)) / 2.0
    width_field = layer_cfg.get("width_field")
    if width_field and width_field in row.index and row[width_field]:
        return float(row[width_field]) / 2.0
    return default_width / 2.0


def _buffered_lines(
    gdf: gpd.GeoDataFrame,
    layer_cfg: dict[str, Any],
    default_width_m: float,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    widths = []
    for _, row in gdf.iterrows():
        if layer_cfg.get("categories") or layer_cfg.get("width_field"):
            widths.append(_road_buffer(layer_cfg, row))
        else:
            widths.append(default_width_m / 2.0)
    gdf = gdf.copy()
    gdf["geometry"] = [
        geom.buffer(width) if geom is not None and not geom.is_empty else geom
        for geom, width in zip(gdf.geometry, widths)
    ]
    return gdf


def _filter_by_field(
    gdf: gpd.GeoDataFrame,
    field: str | None,
    allowed: list[str],
) -> gpd.GeoDataFrame:
    if gdf.empty or not field or field not in gdf.columns:
        return gdf
    allowed_lower = {v.lower() for v in allowed}
    mask = gdf[field].astype(str).str.lower().isin(allowed_lower)
    return gdf[mask].copy()


def build_tile_masks(
    config: dict[str, Any],
    class_rules: dict[str, Any],
    paths: TilePaths,
    *,
    progress: PipelineProgress | None = None,
) -> dict[str, np.ndarray]:
    grid = get_tile_grid(config, paths.tile_id)
    bbox = get_tile_bbox(config, paths.tile_id)
    tlm_paths = config["paths"]["tlm3d"]
    layer_rules = class_rules.get("layers", {})
    mask_values = class_rules["mask_values"]
    priority = class_rules["priority"]

    class_masks: dict[str, np.ndarray] = {}

    def _mask_step(label: str, gdf, mask_name: str, **kwargs) -> None:
        if progress:
            n = 0 if gdf is None else len(gdf)
            progress.substep(f"{label}: {n} features → {mask_name}")
        class_masks[mask_name] = rasterize_layer(gdf, grid, burn_value=1, **kwargs)

    buildings = load_tlm_layer(tlm_paths["buildings"], target_crs=grid.crs, bbox=bbox)
    _mask_step("Buildings", buildings, "building_mask")

    roads_cfg = layer_rules.get("roads", {})
    roads = load_tlm_layer(tlm_paths["roads"], target_crs=grid.crs, bbox=bbox)
    roads_buffered = _buffered_lines(
        roads,
        roads_cfg,
        float(roads_cfg.get("default_width_m", 6.0)),
    )
    if progress:
        progress.substep(f"Roads: {len(roads)} features → road_mask")
    class_masks["road_mask"] = rasterize_layer(roads_buffered, grid, burn_value=1)

    paths_cfg = layer_rules.get("paths", {})
    path_gdf = load_tlm_layer(tlm_paths["paths"], target_crs=grid.crs, bbox=bbox)
    paths_buffered = _buffered_lines(
        path_gdf,
        paths_cfg,
        float(paths_cfg.get("default_width_m", 2.0)),
    )
    if progress:
        progress.substep(f"Paths: {len(path_gdf)} features → path_mask")
    class_masks["path_mask"] = rasterize_layer(paths_buffered, grid, burn_value=1)

    water = load_tlm_layer(tlm_paths["water"], target_crs=grid.crs, bbox=bbox)
    _mask_step("Water", water, "water_mask")

    forest_cfg = layer_rules.get("forest", {})
    forest = load_tlm_layer(tlm_paths["forest"], target_crs=grid.crs, bbox=bbox)
    forest = _filter_by_field(
        forest,
        forest_cfg.get("class_field"),
        forest_cfg.get("forest_values", []),
    )
    if forest.empty:
        forest = load_tlm_layer(tlm_paths["landcover"], target_crs=grid.crs, bbox=bbox)
        forest = _filter_by_field(
            forest,
            forest_cfg.get("class_field"),
            forest_cfg.get("forest_values", []),
        )
    _mask_step("Forest", forest, "forest_mask")

    settlement_cfg = layer_rules.get("settlement", {})
    settlement = load_tlm_layer(tlm_paths["settlement"], target_crs=grid.crs, bbox=bbox)
    settlement = _filter_by_field(
        settlement,
        settlement_cfg.get("class_field"),
        settlement_cfg.get("settlement_values", []),
    )
    _mask_step("Settlement", settlement, "settlement_mask")

    landcover_cfg = layer_rules.get("landcover", {})
    landcover = load_tlm_layer(tlm_paths["landcover"], target_crs=grid.crs, bbox=bbox)
    rock = _filter_by_field(
        landcover,
        landcover_cfg.get("class_field"),
        landcover_cfg.get("rock_values", []),
    )
    _mask_step("Rock/bare ground", rock, "rock_or_bare_ground_mask")

    if paths.dem.exists():
        dem, _ = read_raster(str(paths.dem))
        if dem.ndim == 3:
            dem = dem[0]
        slope = _compute_slope_deg(dem, float(config["resolution_m"]))
        rock_heuristic = class_rules.get("rock_heuristic", {})
        if rock_heuristic.get("combine_with_tlm", True):
            threshold = float(rock_heuristic.get("slope_threshold_deg", 35.0))
            steep = slope > threshold
            if rock_heuristic.get("exclude_forest", True):
                steep &= class_masks["forest_mask"] == 0
            class_masks["rock_or_bare_ground_mask"] = np.maximum(
                class_masks["rock_or_bare_ground_mask"],
                steep.astype(np.uint8),
            )
            if progress:
                progress.substep(
                    f"DEM slope heuristic: {steep.sum()} px added to rock mask "
                    f"(>{threshold}°, excl. forest)"
                )

    occupied = np.zeros((grid.height, grid.width), dtype=bool)
    for name in priority:
        if name in ("open_land_mask", "special_area_mask"):
            continue
        occupied |= class_masks.get(name, np.zeros_like(occupied, dtype=np.uint8)) > 0

    class_masks["open_land_mask"] = (~occupied).astype(np.uint8)
    class_masks["special_area_mask"] = np.zeros((grid.height, grid.width), dtype=np.uint8)

    if progress:
        progress.substep("Deriving open_land_mask and merging priorities → tlm_masks.tif")

    labeled = apply_priority(class_masks, priority, mask_values)
    write_cog(
        str(paths.tlm_masks),
        labeled,
        transform=grid.transform,
        crs=grid.crs,
        nodata=0,
    )

    for name, mask in class_masks.items():
        out_path = paths.intermediate_dir / f"{name}.tif"
        write_cog(
            str(out_path),
            mask.astype(np.uint8),
            transform=grid.transform,
            crs=grid.crs,
            nodata=0,
        )

    return class_masks


def _compute_slope_deg(elevation: np.ndarray, resolution_m: float) -> np.ndarray:
    dz_dy, dz_dx = np.gradient(elevation, resolution_m, resolution_m)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    return np.degrees(slope_rad)
