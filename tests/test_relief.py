import numpy as np

from winter_ortho.rendering.relief import apply_winter_relief, desaturate, luminance


def test_desaturate_reduces_green_dominance():
    rgb = np.zeros((4, 4, 3), dtype=np.float32)
    rgb[..., 0] = 0.3
    rgb[..., 1] = 0.8
    rgb[..., 2] = 0.3
    out = desaturate(rgb, np.full((4, 4), 1.0))
    assert out[..., 1].mean() < rgb[..., 1].mean()
    assert abs(out[..., 0].mean() - out[..., 2].mean()) < 0.05


def test_winter_relief_darkens_shade_and_brightens_sun():
    rgb = np.full((8, 8, 3), 0.8, dtype=np.float32)
    hillshade = np.linspace(0.2, 0.9, 64, dtype=np.float32).reshape(8, 8)
    aspect = np.full((8, 8), 0.0, dtype=np.float32)
    snow_fraction = np.full((8, 8), 0.9, dtype=np.float32)
    snow_color = np.array([0.95, 0.97, 0.99], dtype=np.float32)
    out = apply_winter_relief(
        rgb,
        hillshade=hillshade,
        aspect=aspect,
        snow_fraction=snow_fraction,
        snow_color=snow_color,
        hillshade_strength=0.3,
        aspect_strength=0.0,
        compression=0.5,
    )
    shade_lum = luminance(out)[0, 0]
    sun_lum = luminance(out)[-1, -1]
    assert sun_lum > shade_lum


def test_snow_flattening_reduces_hillshade_on_open_land():
    from winter_ortho.rendering.relief import shade_snow_layer

    hillshade = np.array([[0.2, 0.8]], dtype=np.float32)
    snow_color = np.array([0.95, 0.96, 0.98], dtype=np.float32)
    strong = shade_snow_layer(snow_color, hillshade, 0.2, compression=0.3)
    flat = shade_snow_layer(
        snow_color,
        hillshade,
        0.2,
        compression=0.3,
        snow_fraction=np.array([[0.95, 0.95]], dtype=np.float32),
        snow_flattening=0.9,
    )
    assert np.ptp(luminance(strong)) > np.ptp(luminance(flat))
