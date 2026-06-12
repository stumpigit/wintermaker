from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from winter_ortho.snow_model import rules as snow_rules


@pytest.fixture
def thickness_profile() -> dict:
    return {
        "profile": "test",
        "snow_surface": {"base_snow_height_m": 2.0},
        "elevation": {"reference_m": 1500, "snow_increase_per_100m": 0.03, "max_boost": 0.15},
        "aspect": {"south_thinning": 0.12, "north_boost": 0.08},
        "open_land": {
            "snow_fraction": [0.85, 0.98],
            "snow_brightness": 0.92,
            "snow_texture_strength": 0.35,
        },
        "forest": {
            "snow_fraction": [0.45, 0.72],
            "forest_snow_intensity": 0.68,
            "snow_texture_strength": 0.28,
        },
        "settlement": {
            "snow_fraction": [0.75, 0.92],
            "snow_brightness": 0.88,
            "snow_texture_strength": 0.28,
        },
        "rock": {
            "slope_visibility_threshold_deg": 35,
            "roughness_visibility_threshold": 0.4,
            "max_snow_fraction": 0.62,
            "min_rock_visibility": 0.45,
            "aspect_south_penalty": 0.12,
        },
        "roads": {"snow_fraction": [0.2, 0.5], "road_visibility": 0.85},
        "paths": {"snow_fraction": [0.25, 0.55], "road_visibility": 0.75},
        "buildings": {"roof_snow_intensity": [0.5, 0.8]},
        "water": {"ice_probability": 0.0, "shore_snow_width_px": 0},
    }


def test_rock_burial_uses_neighborhood_thickness(
    monkeypatch: pytest.MonkeyPatch,
    thickness_profile: dict,
) -> None:
    height, width = 32, 32
    class_masks = {
        "building_mask": np.zeros((height, width), dtype=np.uint8),
        "road_mask": np.zeros((height, width), dtype=np.uint8),
        "path_mask": np.zeros((height, width), dtype=np.uint8),
        "water_mask": np.zeros((height, width), dtype=np.uint8),
        "settlement_mask": np.zeros((height, width), dtype=np.uint8),
        "forest_mask": np.zeros((height, width), dtype=np.uint8),
        "rock_or_bare_ground_mask": np.ones((height, width), dtype=np.uint8),
        "open_land_mask": np.zeros((height, width), dtype=np.uint8),
        "special_area_mask": np.zeros((height, width), dtype=np.uint8),
    }
    terrain = {
        "elevation": np.full((height, width), 1800.0, dtype=np.float32),
        "aspect": np.zeros((height, width), dtype=np.float32),
        "terrain_position_index": np.zeros((height, width), dtype=np.float32),
        "hillshade_winter_low_sun": np.full((height, width), 0.6, dtype=np.float32),
        "slope": np.full((height, width), 18.0, dtype=np.float32),
        "roughness": np.full((height, width), 1.2, dtype=np.float32),
    }
    snow_thickness = np.full((height, width), 0.3, dtype=np.float32)
    snow_thickness[10:22, 10:22] = 4.0
    snow_thickness[16, 16] = 0.3
    thickness_profile["rock"]["thickness_burial_radius_m"] = 3
    thickness_profile["rock"]["thickness_burial_factor"] = 0.85

    monkeypatch.setattr(snow_rules, "write_cog", lambda *args, **kwargs: None)
    monkeypatch.setattr(snow_rules, "read_raster", lambda *args, **kwargs: (np.zeros((1, 1)), {}))
    monkeypatch.setattr(
        snow_rules,
        "get_tile_grid",
        lambda *args, **kwargs: SimpleNamespace(
            transform=None, crs="EPSG:2056", width=width, height=height
        ),
    )

    paths = _mock_paths()
    layers = snow_rules.compute_snow_layers(
        {"resolution_m": 1.0},
        thickness_profile,
        paths,
        class_masks,
        terrain,
        snow_thickness=snow_thickness,
    )

    stone = layers["snow_fraction"][16, 16]
    thin_field = layers["snow_fraction"][2, 2]
    assert stone > 0.55
    assert layers["rock_visibility"][16, 16] < layers["rock_visibility"][2, 2]
    assert thin_field < stone


def test_open_land_snow_fraction_follows_thickness(
    monkeypatch: pytest.MonkeyPatch,
    thickness_profile: dict,
) -> None:
    height, width = 16, 16
    class_masks = {
        "building_mask": np.zeros((height, width), dtype=np.uint8),
        "road_mask": np.zeros((height, width), dtype=np.uint8),
        "path_mask": np.zeros((height, width), dtype=np.uint8),
        "water_mask": np.zeros((height, width), dtype=np.uint8),
        "settlement_mask": np.zeros((height, width), dtype=np.uint8),
        "forest_mask": np.zeros((height, width), dtype=np.uint8),
        "rock_or_bare_ground_mask": np.zeros((height, width), dtype=np.uint8),
        "open_land_mask": np.ones((height, width), dtype=np.uint8),
        "special_area_mask": np.zeros((height, width), dtype=np.uint8),
    }
    terrain = {
        "elevation": np.full((height, width), 1800.0, dtype=np.float32),
        "aspect": np.zeros((height, width), dtype=np.float32),
        "terrain_position_index": np.zeros((height, width), dtype=np.float32),
        "hillshade_winter_low_sun": np.full((height, width), 0.6, dtype=np.float32),
        "slope": np.zeros((height, width), dtype=np.float32),
        "roughness": np.zeros((height, width), dtype=np.float32),
    }
    snow_thickness = np.full((height, width), 2.0, dtype=np.float32)
    snow_thickness[0:8, :] = 1.0

    monkeypatch.setattr(snow_rules, "write_cog", lambda *args, **kwargs: None)
    monkeypatch.setattr(snow_rules, "read_raster", lambda *args, **kwargs: (np.zeros((1, 1)), {}))
    monkeypatch.setattr(
        snow_rules,
        "get_tile_grid",
        lambda *args, **kwargs: SimpleNamespace(
            transform=None, crs="EPSG:2056", width=width, height=height
        ),
    )

    paths = _mock_paths()
    layers = snow_rules.compute_snow_layers(
        {},
        thickness_profile,
        paths,
        class_masks,
        terrain,
        snow_thickness=snow_thickness,
    )

    lo, hi = thickness_profile["open_land"]["snow_fraction"]
    thick_pixel = layers["snow_fraction"][10, 10]
    thin_pixel = layers["snow_fraction"][2, 2]
    assert thick_pixel > thin_pixel
    assert thick_pixel >= hi * 0.95
    assert lo + (hi - lo) * 0.45 < thin_pixel < lo + (hi - lo) * 0.65


def test_open_land_steep_slope_reduces_snow_below_minimum(
    monkeypatch: pytest.MonkeyPatch,
    thickness_profile: dict,
) -> None:
    height, width = 16, 16
    class_masks = {
        "building_mask": np.zeros((height, width), dtype=np.uint8),
        "road_mask": np.zeros((height, width), dtype=np.uint8),
        "path_mask": np.zeros((height, width), dtype=np.uint8),
        "water_mask": np.zeros((height, width), dtype=np.uint8),
        "settlement_mask": np.zeros((height, width), dtype=np.uint8),
        "forest_mask": np.zeros((height, width), dtype=np.uint8),
        "rock_or_bare_ground_mask": np.zeros((height, width), dtype=np.uint8),
        "open_land_mask": np.ones((height, width), dtype=np.uint8),
        "special_area_mask": np.zeros((height, width), dtype=np.uint8),
    }
    terrain = {
        "elevation": np.full((height, width), 1800.0, dtype=np.float32),
        "aspect": np.zeros((height, width), dtype=np.float32),
        "terrain_position_index": np.zeros((height, width), dtype=np.float32),
        "hillshade_winter_low_sun": np.full((height, width), 0.6, dtype=np.float32),
        "slope": np.zeros((height, width), dtype=np.float32),
        "roughness": np.zeros((height, width), dtype=np.float32),
    }
    terrain["slope"][0:8, :] = 45.0
    snow_thickness = np.full((height, width), 0.0, dtype=np.float32)
    snow_surface_dem = np.full((height, width), 1800.0, dtype=np.float32)
    thickness_profile["open_land"].update(
        {
            "slope_snow_start_deg": 28,
            "slope_snow_end_deg": 40,
            "slope_min_snow_scale": 0.05,
            "slope_texture_visibility": 0.8,
        }
    )

    monkeypatch.setattr(snow_rules, "write_cog", lambda *args, **kwargs: None)
    monkeypatch.setattr(snow_rules, "read_raster", lambda *args, **kwargs: (np.zeros((1, 1)), {}))
    monkeypatch.setattr(
        snow_rules,
        "get_tile_grid",
        lambda *args, **kwargs: SimpleNamespace(
            transform=None, crs="EPSG:2056", width=width, height=height
        ),
    )

    layers = snow_rules.compute_snow_layers(
        {},
        thickness_profile,
        _mock_paths(),
        class_masks,
        terrain,
        snow_thickness=snow_thickness,
        snow_surface_dem=snow_surface_dem,
    )

    lo = thickness_profile["open_land"]["snow_fraction"][0]
    assert layers["snow_fraction"][0, 0] < lo * 0.2
    assert layers["summer_exposure"][0, 0] > 0.5


def test_open_land_steep_slope_keeps_snow_when_thickness_is_full(
    monkeypatch: pytest.MonkeyPatch,
    thickness_profile: dict,
) -> None:
    height, width = 16, 16
    class_masks = {
        "building_mask": np.zeros((height, width), dtype=np.uint8),
        "road_mask": np.zeros((height, width), dtype=np.uint8),
        "path_mask": np.zeros((height, width), dtype=np.uint8),
        "water_mask": np.zeros((height, width), dtype=np.uint8),
        "settlement_mask": np.zeros((height, width), dtype=np.uint8),
        "forest_mask": np.zeros((height, width), dtype=np.uint8),
        "rock_or_bare_ground_mask": np.zeros((height, width), dtype=np.uint8),
        "open_land_mask": np.ones((height, width), dtype=np.uint8),
        "special_area_mask": np.zeros((height, width), dtype=np.uint8),
    }
    terrain = {
        "elevation": np.full((height, width), 1800.0, dtype=np.float32),
        "aspect": np.zeros((height, width), dtype=np.float32),
        "terrain_position_index": np.zeros((height, width), dtype=np.float32),
        "hillshade_winter_low_sun": np.full((height, width), 0.6, dtype=np.float32),
        "slope": np.full((height, width), 45.0, dtype=np.float32),
        "roughness": np.zeros((height, width), dtype=np.float32),
    }
    snow_thickness = np.full((height, width), 2.0, dtype=np.float32)
    snow_surface_dem = np.full((height, width), 1800.0, dtype=np.float32)
    thickness_profile["open_land"].update(
        {
            "slope_snow_start_deg": 28,
            "slope_snow_end_deg": 40,
            "slope_min_snow_scale": 0.05,
            "slope_snow_strength": 1.0,
            "slope_texture_visibility": 0.8,
        }
    )

    monkeypatch.setattr(snow_rules, "write_cog", lambda *args, **kwargs: None)
    monkeypatch.setattr(snow_rules, "read_raster", lambda *args, **kwargs: (np.zeros((1, 1)), {}))
    monkeypatch.setattr(
        snow_rules,
        "get_tile_grid",
        lambda *args, **kwargs: SimpleNamespace(
            transform=None, crs="EPSG:2056", width=width, height=height
        ),
    )

    layers = snow_rules.compute_snow_layers(
        {},
        thickness_profile,
        _mock_paths(),
        class_masks,
        terrain,
        snow_thickness=snow_thickness,
        snow_surface_dem=snow_surface_dem,
    )

    hi = thickness_profile["open_land"]["snow_fraction"][1]
    assert layers["snow_fraction"][8, 8] >= hi * 0.95
    assert layers["summer_exposure"][8, 8] < 0.05


def test_open_land_slope_snow_strength_zero_disables_steep_penalty(
    monkeypatch: pytest.MonkeyPatch,
    thickness_profile: dict,
) -> None:
    height, width = 16, 16
    class_masks = {
        "building_mask": np.zeros((height, width), dtype=np.uint8),
        "road_mask": np.zeros((height, width), dtype=np.uint8),
        "path_mask": np.zeros((height, width), dtype=np.uint8),
        "water_mask": np.zeros((height, width), dtype=np.uint8),
        "settlement_mask": np.zeros((height, width), dtype=np.uint8),
        "forest_mask": np.zeros((height, width), dtype=np.uint8),
        "rock_or_bare_ground_mask": np.zeros((height, width), dtype=np.uint8),
        "open_land_mask": np.ones((height, width), dtype=np.uint8),
        "special_area_mask": np.zeros((height, width), dtype=np.uint8),
    }
    terrain = {
        "elevation": np.full((height, width), 1800.0, dtype=np.float32),
        "aspect": np.zeros((height, width), dtype=np.float32),
        "terrain_position_index": np.zeros((height, width), dtype=np.float32),
        "hillshade_winter_low_sun": np.full((height, width), 0.6, dtype=np.float32),
        "slope": np.full((height, width), 50.0, dtype=np.float32),
        "roughness": np.zeros((height, width), dtype=np.float32),
    }
    snow_thickness = np.full((height, width), 0.0, dtype=np.float32)
    snow_surface_dem = np.full((height, width), 1800.0, dtype=np.float32)
    lo, hi = thickness_profile["open_land"]["snow_fraction"]
    thickness_profile["open_land"]["slope_snow_strength"] = 0.0

    monkeypatch.setattr(snow_rules, "write_cog", lambda *args, **kwargs: None)
    monkeypatch.setattr(snow_rules, "read_raster", lambda *args, **kwargs: (np.zeros((1, 1)), {}))
    monkeypatch.setattr(
        snow_rules,
        "get_tile_grid",
        lambda *args, **kwargs: SimpleNamespace(
            transform=None, crs="EPSG:2056", width=width, height=height
        ),
    )

    layers = snow_rules.compute_snow_layers(
        {},
        thickness_profile,
        _mock_paths(),
        class_masks,
        terrain,
        snow_thickness=snow_thickness,
        snow_surface_dem=snow_surface_dem,
    )

    assert layers["snow_fraction"][8, 8] == pytest.approx(lo, abs=0.01)
    assert layers["summer_exposure"][8, 8] == pytest.approx(0.0, abs=0.01)


def test_open_land_protrusion_exposes_summer_texture(
    monkeypatch: pytest.MonkeyPatch,
    thickness_profile: dict,
) -> None:
    height, width = 16, 16
    class_masks = {
        "building_mask": np.zeros((height, width), dtype=np.uint8),
        "road_mask": np.zeros((height, width), dtype=np.uint8),
        "path_mask": np.zeros((height, width), dtype=np.uint8),
        "water_mask": np.zeros((height, width), dtype=np.uint8),
        "settlement_mask": np.zeros((height, width), dtype=np.uint8),
        "forest_mask": np.zeros((height, width), dtype=np.uint8),
        "rock_or_bare_ground_mask": np.zeros((height, width), dtype=np.uint8),
        "open_land_mask": np.ones((height, width), dtype=np.uint8),
        "special_area_mask": np.zeros((height, width), dtype=np.uint8),
    }
    dem = np.full((height, width), 1800.0, dtype=np.float32)
    dem[8, 8] = 1801.0
    terrain = {
        "elevation": dem,
        "aspect": np.zeros((height, width), dtype=np.float32),
        "terrain_position_index": np.zeros((height, width), dtype=np.float32),
        "hillshade_winter_low_sun": np.full((height, width), 0.6, dtype=np.float32),
        "slope": np.full((height, width), 12.0, dtype=np.float32),
        "roughness": np.zeros((height, width), dtype=np.float32),
    }
    snow_thickness = np.full((height, width), 2.0, dtype=np.float32)
    snow_surface_dem = np.full((height, width), 1800.0, dtype=np.float32)
    thickness_profile["open_land"].update(
        {
            "protrusion_full_m": 0.5,
            "protrusion_snow_reduction": 0.95,
            "protrusion_texture_visibility": 0.9,
        }
    )

    monkeypatch.setattr(snow_rules, "write_cog", lambda *args, **kwargs: None)
    monkeypatch.setattr(snow_rules, "read_raster", lambda *args, **kwargs: (np.zeros((1, 1)), {}))
    monkeypatch.setattr(
        snow_rules,
        "get_tile_grid",
        lambda *args, **kwargs: SimpleNamespace(
            transform=None, crs="EPSG:2056", width=width, height=height
        ),
    )

    layers = snow_rules.compute_snow_layers(
        {},
        thickness_profile,
        _mock_paths(),
        class_masks,
        terrain,
        snow_thickness=snow_thickness,
        snow_surface_dem=snow_surface_dem,
    )

    lo, hi = thickness_profile["open_land"]["snow_fraction"]
    flat = layers["snow_fraction"][0, 0]
    bump = layers["snow_fraction"][8, 8]
    assert bump < flat
    assert bump < lo + (hi - lo) * 0.5
    assert layers["summer_exposure"][8, 8] > layers["summer_exposure"][0, 0]


def _mock_paths() -> SimpleNamespace:
    paths = SimpleNamespace(tile_id="test", output_dir=SimpleNamespace())
    for name in (
        "snow_fraction",
        "snow_brightness",
        "snow_texture_strength",
        "rock_visibility",
        "forest_snow_intensity",
        "road_visibility",
        "roof_snow_intensity",
        "ice_probability",
        "summer_exposure",
        "landcover_mask",
        "tlm_masks",
    ):
        setattr(paths, name, SimpleNamespace())
    return paths
