from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from winter_ortho.snow_model import rules as snow_rules


@pytest.fixture
def profile() -> dict:
    return {
        "profile": "test",
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


def test_forest_snow_not_overwritten_by_rock(
    monkeypatch: pytest.MonkeyPatch,
    profile: dict,
) -> None:
    height, width = 64, 64
    class_masks = {
        "building_mask": np.zeros((height, width), dtype=np.uint8),
        "road_mask": np.zeros((height, width), dtype=np.uint8),
        "path_mask": np.zeros((height, width), dtype=np.uint8),
        "water_mask": np.zeros((height, width), dtype=np.uint8),
        "settlement_mask": np.zeros((height, width), dtype=np.uint8),
        "forest_mask": np.ones((height, width), dtype=np.uint8),
        "rock_or_bare_ground_mask": np.ones((height, width), dtype=np.uint8),
        "open_land_mask": np.zeros((height, width), dtype=np.uint8),
        "special_area_mask": np.zeros((height, width), dtype=np.uint8),
    }
    terrain = {
        "elevation": np.full((height, width), 1800.0, dtype=np.float32),
        "aspect": np.zeros((height, width), dtype=np.float32),
        "terrain_position_index": np.zeros((height, width), dtype=np.float32),
        "hillshade_winter_low_sun": np.full((height, width), 0.6, dtype=np.float32),
        "slope": np.full((height, width), 45.0, dtype=np.float32),
        "roughness": np.full((height, width), 0.8, dtype=np.float32),
    }

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
    layers = snow_rules.compute_snow_layers({}, profile, paths, class_masks, terrain)

    forest_lo = profile["forest"]["snow_fraction"][0]
    assert layers["snow_fraction"].mean() >= forest_lo
    assert layers["forest_snow_intensity"].mean() > 0.5


def test_settlement_gets_snow(
    monkeypatch: pytest.MonkeyPatch,
    profile: dict,
) -> None:
    height, width = 32, 32
    class_masks = {
        "building_mask": np.zeros((height, width), dtype=np.uint8),
        "road_mask": np.zeros((height, width), dtype=np.uint8),
        "path_mask": np.zeros((height, width), dtype=np.uint8),
        "water_mask": np.zeros((height, width), dtype=np.uint8),
        "settlement_mask": np.ones((height, width), dtype=np.uint8),
        "forest_mask": np.zeros((height, width), dtype=np.uint8),
        "rock_or_bare_ground_mask": np.zeros((height, width), dtype=np.uint8),
        "open_land_mask": np.zeros((height, width), dtype=np.uint8),
        "special_area_mask": np.zeros((height, width), dtype=np.uint8),
    }
    terrain = {
        "elevation": np.full((height, width), 1600.0, dtype=np.float32),
        "aspect": np.zeros((height, width), dtype=np.float32),
        "terrain_position_index": np.zeros((height, width), dtype=np.float32),
        "hillshade_winter_low_sun": np.full((height, width), 0.5, dtype=np.float32),
        "slope": np.zeros((height, width), dtype=np.float32),
        "roughness": np.zeros((height, width), dtype=np.float32),
    }

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
    layers = snow_rules.compute_snow_layers({}, profile, paths, class_masks, terrain)

    assert layers["snow_fraction"].mean() >= profile["settlement"]["snow_fraction"][0]


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
        "landcover_mask",
        "tlm_masks",
    ):
        setattr(paths, name, SimpleNamespace())
    return paths
