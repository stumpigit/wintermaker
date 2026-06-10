from __future__ import annotations

import numpy as np

from winter_ortho.masks.summer_reconcile import compute_summer_appearance, reconcile_masks_with_summer


def _class_rules() -> dict:
    return {
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
        "summer_reconcile": {
            "enabled": True,
            "protect_masks": ["water_mask", "building_mask"],
            "rock": {
                "min_rock_score": 0.38,
                "max_green_excess": 0.055,
                "min_meadow_score": 0.40,
                "min_forest_score": 0.30,
            },
            "forest": {
                "fill_holes": True,
                "hole_close_radius_px": 4,
                "hole_min_forest_score": 0.18,
                "min_forest_score": 0.30,
                "interior_expand_px": 1,
            },
        },
    }


def test_green_summer_removes_false_rock_mask() -> None:
    shape = (32, 32)
    rgb = np.zeros((*shape, 3), dtype=np.float32)
    rgb[..., 1] = 0.45  # meadow green
    masks = {
        "rock_or_bare_ground_mask": np.ones(shape, dtype=np.uint8),
        "forest_mask": np.zeros(shape, dtype=np.uint8),
        "water_mask": np.zeros(shape, dtype=np.uint8),
        "building_mask": np.zeros(shape, dtype=np.uint8),
        "road_mask": np.zeros(shape, dtype=np.uint8),
        "path_mask": np.zeros(shape, dtype=np.uint8),
        "settlement_mask": np.zeros(shape, dtype=np.uint8),
        "open_land_mask": np.zeros(shape, dtype=np.uint8),
        "special_area_mask": np.zeros(shape, dtype=np.uint8),
    }
    out = reconcile_masks_with_summer(rgb, masks, _class_rules())
    assert out["rock_or_bare_ground_mask"].sum() == 0
    assert out["open_land_mask"].sum() > 0


def test_grey_summer_keeps_rock_mask() -> None:
    shape = (16, 16)
    rgb = np.full((*shape, 3), 0.35, dtype=np.float32)
    rgb[..., 0] = 0.38
    rgb[..., 2] = 0.32
    masks = {
        "rock_or_bare_ground_mask": np.ones(shape, dtype=np.uint8),
        "forest_mask": np.zeros(shape, dtype=np.uint8),
        "water_mask": np.zeros(shape, dtype=np.uint8),
        "building_mask": np.zeros(shape, dtype=np.uint8),
        "road_mask": np.zeros(shape, dtype=np.uint8),
        "path_mask": np.zeros(shape, dtype=np.uint8),
        "settlement_mask": np.zeros(shape, dtype=np.uint8),
        "open_land_mask": np.zeros(shape, dtype=np.uint8),
        "special_area_mask": np.zeros(shape, dtype=np.uint8),
    }
    out = reconcile_masks_with_summer(rgb, masks, _class_rules())
    assert out["rock_or_bare_ground_mask"].sum() > 0


def test_forest_hole_filled_when_summer_looks_like_forest() -> None:
    shape = (48, 48)
    rgb = np.zeros((*shape, 3), dtype=np.float32)
    rgb[..., 0] = 0.12
    rgb[..., 1] = 0.28
    rgb[..., 2] = 0.10
    forest = np.zeros(shape, dtype=np.uint8)
    forest[4:44, 4:44] = 1
    forest[20:28, 20:28] = 0  # hole
    masks = {
        "forest_mask": forest,
        "rock_or_bare_ground_mask": np.zeros(shape, dtype=np.uint8),
        "water_mask": np.zeros(shape, dtype=np.uint8),
        "building_mask": np.zeros(shape, dtype=np.uint8),
        "road_mask": np.zeros(shape, dtype=np.uint8),
        "path_mask": np.zeros(shape, dtype=np.uint8),
        "settlement_mask": np.zeros(shape, dtype=np.uint8),
        "open_land_mask": np.zeros(shape, dtype=np.uint8),
        "special_area_mask": np.zeros(shape, dtype=np.uint8),
    }
    scores = compute_summer_appearance(rgb)
    assert scores["forest"][24, 24] > 0.2
    out = reconcile_masks_with_summer(rgb, masks, _class_rules())
    assert out["forest_mask"][24, 24] == 1
