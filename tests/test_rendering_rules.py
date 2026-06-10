from __future__ import annotations

import numpy as np
import pytest

from winter_ortho import pipeline
from winter_ortho.rendering.open_land import render_open_land
from winter_ortho.utils.paths import tile_paths
from winter_ortho.utils.raster import read_raster


@pytest.fixture(autouse=True)
def _chdir(monkeypatch: pytest.MonkeyPatch, synthetic_tile: dict) -> None:
    monkeypatch.chdir(synthetic_tile["root"])


def test_winter_render_is_brighter_than_summer(synthetic_tile: dict) -> None:
    tile_id = synthetic_tile["tile_id"]
    config_path = str(synthetic_tile["config_path"])

    pipeline.run_all(tile_id, "davos", config_path)
    paths = tile_paths(
        __import__("winter_ortho.utils.config", fromlist=["load_config"]).load_config(config_path),
        tile_id,
    )

    summer, _ = read_raster(str(paths.rgb_summer))
    winter, _ = read_raster(str(paths.winter_rgb))
    summer_mean = summer.mean()
    winter_mean = winter.mean()
    assert winter_mean > summer_mean


def test_open_land_heavy_snow_is_mostly_white() -> None:
    """Regression: texture re-blend must scale with snow cover, not stay constant."""
    shape = (32, 32)
    summer = np.zeros((*shape, 3), dtype=np.float32)
    summer[..., 1] = 0.45  # green summer meadow
    mask = np.ones(shape, dtype=np.uint8)
    snow_fraction = np.full(shape, 0.95, dtype=np.float32)
    snow_brightness = np.full(shape, 0.96, dtype=np.float32)
    snow_texture = np.full(shape, 0.3, dtype=np.float32)
    hillshade = np.full(shape, 0.55, dtype=np.float32)
    snow_color = np.array([0.94, 0.96, 0.98], dtype=np.float32)

    result = render_open_land(
        summer,
        mask,
        snow_fraction=snow_fraction,
        snow_brightness=snow_brightness,
        snow_texture_strength=snow_texture,
        hillshade=hillshade,
        snow_color=snow_color,
        hillshade_strength=0.14,
        original_texture_visibility=0.30,
        max_snow_blend=0.95,
        noise_scale_px=4,
    )

    summer_lum = 0.45 * 0.587
    mean_lum = result[..., 0].mean() * 0.299 + result[..., 1].mean() * 0.587 + result[..., 2].mean() * 0.114
    assert mean_lum > 0.82, f"expected near-white snow, got luminance {mean_lum:.2f}"
    assert mean_lum > summer_lum + 0.35


def test_low_snow_brightness_does_not_reduce_cover() -> None:
    """Brightness dims snow color; cover fraction must stay high when snow_fraction is high."""
    shape = (16, 16)
    summer = np.zeros((*shape, 3), dtype=np.float32)
    summer[..., 1] = 0.50
    mask = np.ones(shape, dtype=np.uint8)
    hillshade = np.full(shape, 0.55, dtype=np.float32)
    snow_color = np.array([0.94, 0.96, 0.98], dtype=np.float32)

    bright = render_open_land(
        summer,
        mask,
        snow_fraction=np.full(shape, 0.95, dtype=np.float32),
        snow_brightness=np.full(shape, 0.82, dtype=np.float32),
        snow_texture_strength=np.zeros(shape, dtype=np.float32),
        hillshade=hillshade,
        snow_color=snow_color,
        hillshade_strength=0.14,
        original_texture_visibility=0.08,
        max_snow_blend=0.95,
        noise_scale_px=4,
    )
    mean_g = bright[..., 1].mean()
    assert mean_g > 0.72, f"low brightness must not bleed summer green, got G={mean_g:.2f}"


def test_snow_fraction_in_valid_range(synthetic_tile: dict) -> None:
    tile_id = synthetic_tile["tile_id"]
    config_path = str(synthetic_tile["config_path"])
    pipeline.run_all(tile_id, "davos", config_path)

    paths = tile_paths(
        __import__("winter_ortho.utils.config", fromlist=["load_config"]).load_config(config_path),
        tile_id,
    )
    snow, _ = read_raster(str(paths.snow_fraction))
    valid = snow != -9999.0
    assert snow[valid].min() >= 0.0
    assert snow[valid].max() <= 1.0
