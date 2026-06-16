from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_bounds
from shapely.geometry import LineString, Polygon, box

from winter_ortho.utils.raster import TargetGrid


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "config" / "rendering_profiles").mkdir(parents=True)
    return root


@pytest.fixture
def synthetic_tile(project_root: Path) -> dict:
    tile_id = "test_001"
    bbox = (2792000.0, 1176000.0, 2792100.0, 1176100.0)
    resolution = 1.0
    grid = TargetGrid.from_bbox(bbox, resolution, "EPSG:2056")
    raw = project_root / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "orthophoto").mkdir()
    (raw / "dem").mkdir()
    (raw / "tlm3d").mkdir()

    ortho_path = raw / "orthophoto" / "summer_rgb.tif"
    dem_path = raw / "dem" / "dem_2m.tif"
    _write_rgb(ortho_path, grid)
    _write_dem(dem_path, grid)
    _write_vectors(raw / "tlm3d", bbox)

    config = {
        "crs": "EPSG:2056",
        "tile_size_m": 100,
        "resolution_m": resolution,
        "paths": {
            "raw_root": "data/raw",
            "intermediate_root": "data/intermediate/tiles",
            "output_root": "data/output/tiles",
            "orthophoto": "data/raw/orthophoto/summer_rgb.tif",
            "dem": "data/raw/dem/dem_2m.tif",
            "tlm3d": {
                "buildings": "data/raw/tlm3d/buildings.gpkg",
                "roads": "data/raw/tlm3d/roads.gpkg",
                "paths": "data/raw/tlm3d/paths.gpkg",
                "water": "data/raw/tlm3d/water.gpkg",
                "forest": "data/raw/tlm3d/forest.gpkg",
                "settlement": "data/raw/tlm3d/settlement.gpkg",
                "landcover": "data/raw/tlm3d/landcover.gpkg",
            },
            "winter_reference": "data/raw/reference/winter_rgb.tif",
        },
        "tiles": {tile_id: {"bbox": list(bbox)}},
        "harmonize": {
            "orthophoto_resampling": "bilinear",
            "dem_resampling": "bilinear",
        },
        "terrain": {
            "hillshade": {"azimuth": 150, "altitude": 25},
            "tpi_radius_px": 3,
            "roughness_radius_px": 2,
            "flow_iterations": 10,
        },
        "qa": {
            "building_edge_tolerance": 0.35,
            "road_brightness_min": 0.15,
            "water_boundary_tolerance": 0.25,
            "forest_boundary_tolerance": 0.40,
            "hallucination_tolerance": 0.60,
        },
    }
    class_rules = {
        "priority": [
            "building_mask",
            "road_mask",
            "path_mask",
            "water_mask",
            "settlement_mask",
            "forest_mask",
            "rock_or_bare_ground_mask",
            "open_land_mask",
            "special_area_mask",
        ],
        "mask_values": {
            "building_mask": 1,
            "road_mask": 2,
            "path_mask": 3,
            "water_mask": 4,
            "settlement_mask": 5,
            "forest_mask": 6,
            "rock_or_bare_ground_mask": 7,
            "open_land_mask": 8,
            "special_area_mask": 9,
        },
        "layers": {
            "roads": {"default_width_m": 6.0},
            "paths": {"default_width_m": 2.0},
            "forest": {"class_field": "objektart", "forest_values": ["Wald"]},
            "settlement": {"class_field": "objektart", "settlement_values": ["Siedlung"]},
            "landcover": {
                "class_field": "objektart",
                "rock_values": ["Fels"],
                "open_land_values": ["Wiese"],
            },
        },
        "rock_heuristic": {
            "slope_threshold_deg": 35.0,
            "combine_with_tlm": True,
            "exclude_forest": True,
        },
        "protect_hard_masks": ["water_mask", "building_mask", "road_mask", "forest_mask"],
    }
    davos_profile = {
        "profile": "default",
        "elevation": {"reference_m": 1560, "snow_increase_per_100m": 0.03, "max_boost": 0.15},
        "aspect": {"south_thinning": 0.12, "north_boost": 0.08},
        "open_land": {
            "snow_fraction": [0.78, 0.92],
            "snow_brightness": 0.82,
            "snow_texture_strength": 0.32,
            "original_texture_visibility": 0.32,
            "max_snow_blend": 0.68,
            "detail_preservation": 0.58,
        },
        "forest": {
            "snow_fraction": [0.62, 0.85],
            "forest_snow_intensity": 0.82,
            "original_texture_visibility": 0.28,
            "contrast_reduction": 0.08,
            "snow_texture_strength": 0.28,
            "green_suppression": 0.50,
            "winter_luminance_lo": 0.46,
            "winter_luminance_hi": 0.76,
            "crown_highlight_strength": 0.40,
        },
        "settlement": {
            "snow_fraction": [0.75, 0.92],
            "snow_brightness": 0.88,
            "snow_texture_strength": 0.28,
        },
        "rock": {
            "slope_visibility_threshold_deg": 35,
            "roughness_visibility_threshold": 0.4,
            "max_snow_fraction": 0.78,
            "min_rock_visibility": 0.40,
            "aspect_south_penalty": 0.10,
            "gentle_slope_max_deg": 28,
            "steep_slope_min_deg": 42,
            "gentle_snow_boost": 0.28,
            "gentle_render_boost": 0.20,
        },
        "roads": {"snow_fraction": [0.2, 0.5], "road_visibility": 0.85, "gray_white_mix": 0.7},
        "paths": {
            "snow_fraction": [0.90, 0.98],
            "snow_brightness": 0.94,
            "snow_texture_strength": 0.42,
            "bury_strength": 0.98,
        },
        "buildings": {
            "roof_snow_intensity": [0.5, 0.8],
            "edge_preservation": 0.95,
            "brighten_factor": 0.55,
        },
        "water": {
            "protected": True,
            "ice_probability": 0.0,
            "shore_snow_width_px": 2,
            "shore_snow_intensity": 0.4,
            "water_darken": 0.85,
        },
        "rendering": {
            "snow_color": [0.94, 0.96, 0.98],
            "road_color": [0.78, 0.80, 0.82],
            "noise_scale_px": 4,
            "hillshade_strength_open_land": 0.10,
            "hillshade_strength_rock": 0.16,
            "hillshade_strength_settlement": 0.12,
            "relief": {
                "hillshade_strength": 0.09,
                "aspect_strength": 0.05,
                "compression": 0.30,
                "min_snow": 0.35,
                "class_weights": {"open_land": 0.12, "rock": 0.58},
            },
        },
    }

    (project_root / "config").mkdir(exist_ok=True)
    (project_root / "config" / "default.yaml").write_text(
        yaml.dump(config), encoding="utf-8"
    )
    (project_root / "config" / "class_rules.yaml").write_text(
        yaml.dump(class_rules), encoding="utf-8"
    )
    (project_root / "config" / "rendering_profiles" / "default.yaml").write_text(
        yaml.dump(davos_profile), encoding="utf-8"
    )

    return {
        "root": project_root,
        "tile_id": tile_id,
        "config_path": project_root / "config" / "default.yaml",
        "grid": grid,
    }


def _write_rgb(path: Path, grid: TargetGrid) -> None:
    y, x = np.mgrid[0 : grid.height, 0 : grid.width]
    r = np.clip(80 + x * 2, 0, 255).astype(np.uint8)
    g = np.clip(120 + y, 0, 255).astype(np.uint8)
    b = np.clip(60 + (x + y), 0, 255).astype(np.uint8)
    data = np.stack([r, g, b])
    _write_raster(path, data, grid, nodata=0)


def _write_dem(path: Path, grid: TargetGrid) -> None:
    y, x = np.mgrid[0 : grid.height, 0 : grid.width]
    elevation = 1500 + x * 0.5 + y * 0.3
    _write_raster(path, elevation[np.newaxis, ...].astype(np.float32), grid, nodata=-9999.0)


def _write_raster(path: Path, data: np.ndarray, grid: TargetGrid, nodata: float) -> None:
    count, height, width = data.shape
    profile = {
        "driver": "GTiff",
        "dtype": data.dtype,
        "width": width,
        "height": height,
        "count": count,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def _write_vectors(tlm_dir: Path, bbox: tuple[float, float, float, float]) -> None:
    minx, miny, maxx, maxy = bbox
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2

    buildings = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Polygon([(cx - 10, cy - 10), (cx + 10, cy - 10), (cx + 10, cy + 10), (cx - 10, cy + 10)])],
        crs="EPSG:2056",
    )
    roads = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[LineString([(minx + 5, miny + 5), (maxx - 5, maxy - 5)])],
        crs="EPSG:2056",
    )
    paths = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[LineString([(minx + 10, maxy - 10), (maxx - 10, miny + 10)])],
        crs="EPSG:2056",
    )
    water = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(minx + 60, miny + 60, minx + 90, miny + 90)],
        crs="EPSG:2056",
    )
    forest = gpd.GeoDataFrame(
        {"objektart": ["Wald"]},
        geometry=[box(minx + 5, miny + 50, minx + 40, miny + 90)],
        crs="EPSG:2056",
    )
    settlement = gpd.GeoDataFrame(
        {"objektart": ["Siedlung"]},
        geometry=[box(cx - 20, cy - 20, cx + 20, cy + 20)],
        crs="EPSG:2056",
    )
    landcover = gpd.GeoDataFrame(
        {"objektart": ["Fels", "Wiese"]},
        geometry=[
            box(maxx - 40, maxy - 40, maxx - 5, maxy - 5),
            box(minx + 40, miny + 5, maxx - 40, miny + 40),
        ],
        crs="EPSG:2056",
    )

    buildings.to_file(tlm_dir / "buildings.gpkg", driver="GPKG")
    roads.to_file(tlm_dir / "roads.gpkg", driver="GPKG")
    paths.to_file(tlm_dir / "paths.gpkg", driver="GPKG")
    water.to_file(tlm_dir / "water.gpkg", driver="GPKG")
    forest.to_file(tlm_dir / "forest.gpkg", driver="GPKG")
    settlement.to_file(tlm_dir / "settlement.gpkg", driver="GPKG")
    landcover.to_file(tlm_dir / "landcover.gpkg", driver="GPKG")
