import numpy as np

from winter_ortho.rendering.relief import apply_map_shading, apply_winter_relief, desaturate, luminance


def test_desaturate_reduces_green_dominance():
    rgb = np.zeros((4, 4, 3), dtype=np.float32)
    rgb[..., 0] = 0.3
    rgb[..., 1] = 0.8
    rgb[..., 2] = 0.3
    out = desaturate(rgb, np.full((4, 4), 1.0))
    assert out[..., 1].mean() < rgb[..., 1].mean()
    assert abs(out[..., 0].mean() - out[..., 2].mean()) < 0.05


def test_winter_relief_attenuates_summer_texture_on_deep_snow():
    rgb = np.full((8, 8, 3), 0.8, dtype=np.float32)
    hillshade = np.full((8, 8), 0.5, dtype=np.float32)
    aspect = np.full((8, 8), 0.0, dtype=np.float32)
    snow_color = np.array([0.95, 0.97, 0.99], dtype=np.float32)
    summer = np.zeros((8, 8, 3), dtype=np.float32)
    summer[..., 0] = np.linspace(0.2, 0.9, 64, dtype=np.float32).reshape(8, 8)
    summer[..., 1] = summer[..., 0]
    summer[..., 2] = summer[..., 0]
    common = dict(
        hillshade=hillshade,
        aspect=aspect,
        snow_color=snow_color,
        summer_rgb=summer,
        hillshade_strength=0.22,
        summer_relief_weight=0.30,
        min_snow=0.30,
        snow_tint_strength=0.0,
    )

    deep = apply_winter_relief(
        rgb,
        snow_fraction=np.full((8, 8), 0.96, dtype=np.float32),
        **common,
    )
    shallow = apply_winter_relief(
        rgb,
        snow_fraction=np.full((8, 8), 0.55, dtype=np.float32),
        **common,
    )
    assert np.ptp(luminance(deep)) < np.ptp(luminance(shallow))


def test_winter_relief_darkens_shade_and_brightens_sun():
    rgb = np.full((8, 8, 3), 0.8, dtype=np.float32)
    hillshade = np.linspace(0.2, 0.9, 64, dtype=np.float32).reshape(8, 8)
    aspect = np.full((8, 8), 0.0, dtype=np.float32)
    snow_fraction = np.full((8, 8), 0.9, dtype=np.float32)
    snow_color = np.array([0.95, 0.97, 0.99], dtype=np.float32)
    summer = np.full((8, 8, 3), 0.6, dtype=np.float32)
    out = apply_winter_relief(
        rgb,
        hillshade=hillshade,
        aspect=aspect,
        snow_fraction=snow_fraction,
        snow_color=snow_color,
        summer_rgb=summer,
        hillshade_strength=0.3,
        aspect_strength=0.0,
        compression=0.5,
        summer_relief_weight=0.4,
    )
    shade_lum = luminance(out)[0, 0]
    sun_lum = luminance(out)[-1, -1]
    assert sun_lum > shade_lum


def test_map_shading_darkens_south_slopes():
    rgb = np.full((32, 32, 3), 0.82, dtype=np.float32)
    aspect = np.full((32, 32), 180.0, dtype=np.float32)
    aspect[:16, :] = 0.0
    hill = np.full((32, 32), 0.5, dtype=np.float32)
    hill[:16, :] = 0.72
    hill[16:, :] = 0.28
    snow_fraction = np.full((32, 32), 0.95, dtype=np.float32)
    slope = np.full((32, 32), 18.0, dtype=np.float32)
    out = apply_map_shading(
        rgb,
        hillshade_generalized=hill,
        aspect=aspect,
        slope=slope,
        snow_fraction=snow_fraction,
        sun_azimuth=45.0,
        strength=0.6,
        south_shadow_boost=1.35,
    )
    north_lum = luminance(out)[:16].mean()
    south_lum = luminance(out)[16:].mean()
    assert south_lum < north_lum


def test_ne_sun_darkens_south_slopes():
    rgb = np.full((4, 4, 3), 0.85, dtype=np.float32)
    hillshade = np.full((4, 4), 0.5, dtype=np.float32)
    snow_fraction = np.full((4, 4), 0.95, dtype=np.float32)
    snow_color = np.array([0.96, 0.98, 0.99], dtype=np.float32)
    north = apply_winter_relief(
        rgb,
        hillshade=hillshade,
        aspect=np.full((4, 4), 0.0, dtype=np.float32),
        snow_fraction=snow_fraction,
        snow_color=snow_color,
        hillshade_strength=0.0,
        aspect_strength=0.25,
        sun_azimuth=45.0,
        summer_relief_weight=0.0,
    )
    south = apply_winter_relief(
        rgb,
        hillshade=hillshade,
        aspect=np.full((4, 4), 180.0, dtype=np.float32),
        snow_fraction=snow_fraction,
        snow_color=snow_color,
        hillshade_strength=0.0,
        aspect_strength=0.25,
        sun_azimuth=45.0,
        south_shadow_boost=1.3,
        summer_relief_weight=0.0,
    )
    assert luminance(south).mean() < luminance(north).mean()


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
