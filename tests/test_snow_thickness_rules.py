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
    snow_thickness = np.full((height, width), 0.8, dtype=np.float32)
    snow_thickness[0:8, :] = 0.2

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
    assert lo < thin_pixel < hi


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
    snow_thickness = np.full((height, width), 0.25, dtype=np.float32)
    snow_surface_dem = np.full((height, width), 1800.0, dtype=np.float32)
    thickness_profile["open_land"].update(
        {
            "full_snow_thickness_m": 0.5,
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


def test_open_land_protrusion_full_white_when_thickness_exceeds_threshold(
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
    dem[8, 8] = 1802.0
    terrain = {
        "elevation": dem,
        "aspect": np.zeros((height, width), dtype=np.float32),
        "terrain_position_index": np.zeros((height, width), dtype=np.float32),
        "hillshade_winter_low_sun": np.full((height, width), 0.6, dtype=np.float32),
        "slope": np.full((height, width), 38.0, dtype=np.float32),
        "roughness": np.zeros((height, width), dtype=np.float32),
    }
    snow_thickness = np.full((height, width), 0.8, dtype=np.float32)
    snow_surface_dem = np.full((height, width), 1800.0, dtype=np.float32)
    thickness_profile["open_land"].update(
        {
            "full_snow_thickness_m": 0.5,
            "protrusion_full_m": 0.5,
            "protrusion_strength": 0.85,
            "protrusion_snow_reduction": 0.90,
            "protrusion_texture_visibility": 0.75,
            "slope_snow_strength": 0.75,
            "slope_snow_start_deg": 32,
            "slope_snow_end_deg": 44,
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


def test_open_land_steep_non_accumulation_uses_blanket_depth_for_gating(
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
    snow_thickness = np.zeros((height, width), dtype=np.float32)
    blanket_thickness = np.full((height, width), 2.0, dtype=np.float32)
    accumulation_mask = np.zeros((height, width), dtype=np.uint8)
    snow_surface_dem = terrain["elevation"].copy()
    thickness_profile["open_land"].update(
        {
            "full_snow_thickness_m": 0.5,
            "slope_snow_start_deg": 32,
            "slope_snow_end_deg": 44,
            "slope_min_snow_scale": 0.35,
            "slope_snow_strength": 0.5,
            "slope_min_snow_fraction": 0.96,
            "deck_depth_cover_floor": 0.55,
            "deck_snow_fraction_boost": 0.85,
            "slope_texture_visibility": 0.55,
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
        blanket_thickness=blanket_thickness,
        accumulation_mask=accumulation_mask,
        snow_surface_dem=snow_surface_dem,
    )

    hi = thickness_profile["open_land"]["snow_fraction"][1]
    min_steep = thickness_profile["open_land"]["slope_min_snow_fraction"]
    assert layers["snow_fraction"][8, 8] >= min_steep * 0.98
    assert layers["snow_fraction"][8, 8] <= hi
    assert layers["summer_exposure"][8, 8] < 0.05


def test_open_land_transition_band_avoids_summer_exposure_with_deck_weight(
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
    dem += np.random.default_rng(3).normal(0, 0.4, dem.shape).astype(np.float32)
    terrain = {
        "elevation": dem,
        "aspect": np.zeros((height, width), dtype=np.float32),
        "terrain_position_index": np.zeros((height, width), dtype=np.float32),
        "hillshade_winter_low_sun": np.full((height, width), 0.6, dtype=np.float32),
        "slope": np.full((height, width), 37.0, dtype=np.float32),
        "roughness": np.zeros((height, width), dtype=np.float32),
    }
    snow_thickness = np.full((height, width), 0.12, dtype=np.float32)
    blanket_thickness = np.full((height, width), 1.8, dtype=np.float32)
    accumulation_mask = np.zeros((height, width), dtype=np.uint8)
    snow_cover_weight = np.full((height, width), 0.45, dtype=np.float32)
    snow_surface_dem = (dem * 0.55 + (dem + 0.8) * 0.45).astype(np.float32)
    thickness_profile["open_land"].update(
        {
            "full_snow_thickness_m": 0.5,
            "protrusion_full_m": 0.5,
            "protrusion_strength": 0.85,
            "protrusion_snow_reduction": 0.90,
            "protrusion_texture_visibility": 0.75,
            "slope_min_snow_fraction": 0.96,
            "deck_depth_cover_floor": 0.55,
            "deck_snow_fraction_boost": 0.85,
            "slope_snow_strength": 0.4,
            "slope_snow_start_deg": 28,
            "slope_snow_end_deg": 48,
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
        blanket_thickness=blanket_thickness,
        accumulation_mask=accumulation_mask,
        snow_surface_dem=snow_surface_dem,
        snow_cover_weight=snow_cover_weight,
    )

    min_steep = thickness_profile["open_land"]["slope_min_snow_fraction"]
    assert layers["snow_fraction"][8, 8] >= min_steep * 0.98
    assert layers["summer_exposure"][8, 8] < 0.05


def test_open_land_depth_gating_respects_zero_cover_and_geometric_cap() -> None:
    snow_thickness = np.array(
        [[0.0, 0.15, 4.0], [0.0, 1.5, 9.5]],
        dtype=np.float32,
    )
    blanket = np.full((2, 3), 10.0, dtype=np.float32)
    cover = np.array([[0.0, 0.17, 1.0], [0.0, 0.55, 1.0]], dtype=np.float32)
    mask = np.ones((2, 3), dtype=bool)

    depth = snow_rules._open_land_depth_for_gating(
        mask,
        snow_thickness_m=snow_thickness,
        blanket_thickness_m=blanket,
        accumulation_mask=None,
        snow_cover_weight=cover,
        deck_depth_cover_floor=0.72,
    )

    assert depth.shape == (6,)
    assert depth[0] == 0.0
    assert depth[1] == pytest.approx(0.15, abs=0.01)
    assert depth[2] == pytest.approx(4.0, abs=0.05)
    assert depth[4] == pytest.approx(1.5, abs=0.05)


def test_open_land_blanket_thickness_gates_protrusion_when_geometric_is_zero(
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
    dem[8, 8] = 1802.0
    terrain = {
        "elevation": dem,
        "aspect": np.zeros((height, width), dtype=np.float32),
        "terrain_position_index": np.zeros((height, width), dtype=np.float32),
        "hillshade_winter_low_sun": np.full((height, width), 0.6, dtype=np.float32),
        "slope": np.full((height, width), 18.0, dtype=np.float32),
        "roughness": np.zeros((height, width), dtype=np.float32),
    }
    snow_thickness = np.zeros((height, width), dtype=np.float32)
    blanket_thickness = np.full((height, width), 0.8, dtype=np.float32)
    accumulation_mask = np.ones((height, width), dtype=np.uint8)
    snow_surface_dem = np.full((height, width), 1800.0, dtype=np.float32)
    thickness_profile["open_land"].update(
        {
            "full_snow_thickness_m": 0.5,
            "protrusion_full_m": 0.5,
            "protrusion_strength": 0.85,
            "protrusion_snow_reduction": 0.90,
            "protrusion_texture_visibility": 0.75,
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
        blanket_thickness=blanket_thickness,
        accumulation_mask=accumulation_mask,
        snow_surface_dem=snow_surface_dem,
    )

    hi = thickness_profile["open_land"]["snow_fraction"][1]
    assert layers["snow_fraction"][8, 8] >= hi * 0.95
    assert layers["summer_exposure"][8, 8] < 0.05


def test_rock_blanket_thickness_yields_high_snow_fraction_on_protrusion(
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
        "rock_or_bare_ground_mask": np.ones((height, width), dtype=np.uint8),
        "open_land_mask": np.zeros((height, width), dtype=np.uint8),
        "special_area_mask": np.zeros((height, width), dtype=np.uint8),
    }
    terrain = {
        "elevation": np.full((height, width), 1800.0, dtype=np.float32),
        "aspect": np.zeros((height, width), dtype=np.float32),
        "terrain_position_index": np.zeros((height, width), dtype=np.float32),
        "hillshade_winter_low_sun": np.full((height, width), 0.6, dtype=np.float32),
        "slope": np.full((height, width), 22.0, dtype=np.float32),
        "roughness": np.full((height, width), 1.0, dtype=np.float32),
    }
    snow_thickness = np.zeros((height, width), dtype=np.float32)
    blanket_thickness = np.full((height, width), 0.8, dtype=np.float32)
    accumulation_mask = np.ones((height, width), dtype=np.uint8)
    thickness_profile["rock"].update(
        {
            "full_snow_thickness_m": 0.5,
            "max_snow_fraction": 0.94,
            "thickness_burial_factor": 0.85,
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
        {"resolution_m": 1.0},
        thickness_profile,
        _mock_paths(),
        class_masks,
        terrain,
        snow_thickness=snow_thickness,
        blanket_thickness=blanket_thickness,
        accumulation_mask=accumulation_mask,
    )

    max_snow = thickness_profile["rock"]["max_snow_fraction"]
    assert layers["snow_fraction"][8, 8] >= max_snow * 0.90
    assert layers["rock_visibility"][8, 8] < 0.25


def test_steep_rock_keeps_visibility_under_full_blanket(
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
        "rock_or_bare_ground_mask": np.ones((height, width), dtype=np.uint8),
        "open_land_mask": np.zeros((height, width), dtype=np.uint8),
        "special_area_mask": np.zeros((height, width), dtype=np.uint8),
    }
    terrain = {
        "elevation": np.full((height, width), 1800.0, dtype=np.float32),
        "aspect": np.zeros((height, width), dtype=np.float32),
        "terrain_position_index": np.zeros((height, width), dtype=np.float32),
        "hillshade_winter_low_sun": np.full((height, width), 0.6, dtype=np.float32),
        "slope": np.full((height, width), 52.0, dtype=np.float32),
        "roughness": np.full((height, width), 1.0, dtype=np.float32),
    }
    blanket_thickness = np.full((height, width), 8.0, dtype=np.float32)
    accumulation_mask = np.ones((height, width), dtype=np.uint8)
    thickness_profile["rock"].update(
        {
            "full_snow_thickness_m": 8.0,
            "max_snow_fraction": 0.94,
            "thickness_burial_factor": 0.85,
            "gentle_slope_max_deg": 32,
            "steep_slope_min_deg": 46,
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
        {"resolution_m": 1.0},
        thickness_profile,
        _mock_paths(),
        class_masks,
        terrain,
        snow_thickness=blanket_thickness,
        blanket_thickness=blanket_thickness,
        accumulation_mask=accumulation_mask,
    )

    assert layers["rock_visibility"][8, 8] > 0.35


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
