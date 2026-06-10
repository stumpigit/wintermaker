import numpy as np

from winter_ortho.features.terrain import compute_generalized_hillshade
from winter_ortho.rendering.summer_structure import (
    apply_summer_anchored_grade,
    apply_summer_cast_shadows,
    remap_luminance_structure,
    soft_highlight_rolloff,
    summer_cast_shadow_field,
    summer_luminance_map,
)


def test_summer_luminance_map_tracks_shadows():
    rgb = np.zeros((16, 16, 3), dtype=np.float32)
    rgb[:8, :, 1] = 0.2
    rgb[8:, :, 1] = 0.7
    field = summer_luminance_map(rgb)
    assert field[:8].mean() < field[8:].mean()


def test_remap_follows_summer_layout():
    summer = np.zeros((16, 16, 3), dtype=np.float32)
    summer[..., 1] = np.linspace(0.1, 0.8, 16 * 16).reshape(16, 16)
    winter = np.full((16, 16, 3), 0.9, dtype=np.float32)
    mask = np.ones((16, 16), dtype=np.uint8)
    out = remap_luminance_structure(
        winter,
        summer,
        1.0,
        target_lo=0.2,
        target_hi=0.8,
        mask=mask,
    )
    lum = 0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]
    assert lum[0, 0] < lum[-1, -1]


def test_highlight_rolloff_preserves_shadows():
    rgb = np.zeros((8, 8, 3), dtype=np.float32)
    rgb[..., 0] = np.linspace(0.2, 0.95, 64).reshape(8, 8)
    out = soft_highlight_rolloff(rgb, knee=0.84, compression=0.5)
    lum = 0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]
    assert lum[0, 0] < 0.35
    assert lum[-1, -1] <= rgb[..., 0][-1, -1]


def test_cast_shadow_darkens_snow_near_buildings() -> None:
    shape = (64, 64)
    summer = np.full((*shape, 3), 0.55, dtype=np.float32)
    summer[20:30, 20:30, :] = 0.15
    building = np.zeros(shape, dtype=np.uint8)
    building[24:28, 24:28] = 1
    cast = summer_cast_shadow_field(summer, building_mask=building, forest_mask=None)
    winter = np.full((*shape, 3), 0.9, dtype=np.float32)
    snow_fraction = np.full(shape, 0.95, dtype=np.float32)
    out = apply_summer_cast_shadows(winter, cast, snow_fraction=snow_fraction, strength=0.8)
    lum = 0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]
    assert lum[22, 22] < lum[50, 50]


def test_generalized_hillshade_is_smoother_than_fine() -> None:
    y, x = np.mgrid[0:64, 0:64]
    elevation = (np.sin(x / 6.0) * 4 + np.sin(y / 3.0) * 1.5).astype(np.float32)
    terrain_cfg = {"generalized_hillshade_sigma_m": 40.0, "hillshade": {"azimuth": 150, "altitude": 25}}
    fine = compute_generalized_hillshade(elevation, 2.0, {**terrain_cfg, "generalized_hillshade_sigma_m": 0.5})
    macro = compute_generalized_hillshade(elevation, 2.0, terrain_cfg)
    assert macro.std() < fine.std()


def test_grade_follows_summer_globally():
    summer = np.zeros((32, 32, 3), dtype=np.float32)
    summer[..., 1] = np.linspace(0.1, 0.8, 32 * 32).reshape(32, 32)
    winter = np.full((32, 32, 3), 0.9, dtype=np.float32)
    masks = {"water_mask": np.zeros((32, 32), dtype=np.uint8)}
    masks["open_land_mask"] = np.ones((32, 32), dtype=np.uint8)
    cfg = {
        "highlight_rolloff": False,
        "protect_masks": ["water_mask"],
        "classes": {
            "open_land_mask": {"lum_lo": 0.15, "lum_hi": 0.85, "strength": 0.9},
        },
    }
    out = apply_summer_anchored_grade(winter, summer, masks, cfg)
    lum = 0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]
    assert lum[0, 0] < lum[-1, -1]
