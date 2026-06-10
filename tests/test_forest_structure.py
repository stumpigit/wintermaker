import numpy as np

from winter_ortho.rendering.forest import render_forest


def test_forest_preserves_crown_contrast():
    h, w = 64, 64
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[8:56, 8:56] = 1

    y, x = np.mgrid[0:h, 0:w]
    crowns = (np.sin(x * 0.55) * np.sin(y * 0.55) + 1.0) * 0.5
    rgb[..., 0] = 0.15 + crowns * 0.12
    rgb[..., 1] = 0.35 + crowns * 0.25
    rgb[..., 2] = 0.12 + crowns * 0.10

    snow_fraction = np.full((h, w), 0.85, dtype=np.float32)
    forest_snow_intensity = np.full((h, w), 0.9, dtype=np.float32)
    hillshade = np.full((h, w), 0.55, dtype=np.float32)

    out = render_forest(
        rgb,
        mask,
        snow_fraction=snow_fraction,
        forest_snow_intensity=forest_snow_intensity,
        snow_color=np.array([0.94, 0.96, 0.98], dtype=np.float32),
        hillshade=hillshade,
        contrast_reduction=0.1,
        original_texture_visibility=0.08,
        crown_tophat_radius_px=4,
    )
    active = mask > 0
    lum = 0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]
    assert lum[active].std() > 0.03
    assert lum[active].max() - lum[active].min() > 0.10
    assert np.percentile(lum[active], 10) < np.percentile(lum[active], 90)
    assert np.percentile(lum[active], 90) < 0.82


def test_forest_does_not_preserve_summer_green_hue():
    h, w = 32, 32
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    mask = np.ones((h, w), dtype=np.uint8)
    rgb[..., 0] = 0.18
    rgb[..., 1] = 0.48
    rgb[..., 2] = 0.14

    out = render_forest(
        rgb,
        mask,
        snow_fraction=np.full((h, w), 0.9, dtype=np.float32),
        forest_snow_intensity=np.full((h, w), 0.9, dtype=np.float32),
        snow_color=np.array([0.94, 0.96, 0.98], dtype=np.float32),
        hillshade=np.full((h, w), 0.55, dtype=np.float32),
        contrast_reduction=0.1,
        original_texture_visibility=0.08,
    )
    summer_gr = rgb[..., 1].mean() / max(rgb[..., 0].mean(), 1e-4)
    winter_gr = out[..., 1].mean() / max(out[..., 0].mean(), 1e-4)
    assert winter_gr < summer_gr - 0.05
