from __future__ import annotations

import numpy as np
from scipy import ndimage


def luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )


def desaturate(rgb: np.ndarray, strength: np.ndarray) -> np.ndarray:
    strength = np.clip(strength, 0.0, 1.0)
    if strength.ndim == 2:
        strength = strength[..., np.newaxis]
    lum = luminance(rgb)[..., np.newaxis]
    return rgb * (1.0 - strength) + lum * strength


def aspect_relief_for_sun(
    aspect_deg: np.ndarray,
    sun_azimuth_deg: float,
    strength: float,
    *,
    shadow_boost: float = 1.0,
) -> np.ndarray:
    """
    Map-style aspect shading: bright slopes facing the sun, dark on the lee side.

    aspect_deg is downslope direction (0°=N, 90°=E, 180°=S). sun_azimuth_deg is
    the illumination azimuth (e.g. 45° = north-east).
    """
    facing = np.cos(np.radians(aspect_deg - sun_azimuth_deg))
    if shadow_boost != 1.0:
        facing = np.where(facing < 0.0, facing * shadow_boost, facing)
    return facing * float(strength)


def compress_hillshade(hillshade: np.ndarray, compression: float) -> np.ndarray:
    """Reduce hillshade dynamic range — deep snow blankets small terrain undulation."""
    compression = float(np.clip(compression, 0.0, 1.0))
    return (0.5 + (hillshade - 0.5) * compression).astype(np.float32)


def shade_snow_layer(
    snow_color: np.ndarray,
    hillshade: np.ndarray,
    strength: float,
    *,
    compression: float = 1.0,
    snow_fraction: np.ndarray | None = None,
    snow_flattening: float = 0.0,
    shade_field: np.ndarray | None = None,
    shadow_boost: float = 1.0,
    highlight_cap: float = 1.0,
) -> np.ndarray:
    field = shade_field if shade_field is not None else compress_hillshade(hillshade, compression)
    effective = float(strength)
    if snow_fraction is not None and snow_flattening > 0.0:
        effective = effective * (1.0 - np.clip(snow_fraction, 0.0, 1.0) * snow_flattening)

    delta = field - 0.5
    neg = np.minimum(delta, 0.0) * float(shadow_boost)
    pos = np.maximum(delta, 0.0) * float(highlight_cap)
    relief = neg + pos

    layer = np.empty((*field.shape, 3), dtype=np.float32)
    layer[:] = snow_color
    if np.ndim(effective) == 0:
        shade = relief[..., np.newaxis] * effective
    else:
        shade = relief[..., np.newaxis] * effective[..., np.newaxis]
    return np.clip(layer + shade, 0.0, 1.0)


def apply_map_shading(
    rgb: np.ndarray,
    *,
    hillshade_generalized: np.ndarray,
    aspect: np.ndarray,
    slope: np.ndarray,
    snow_fraction: np.ndarray,
    sun_azimuth: float,
    strength: float = 0.50,
    hillshade_weight: float = 0.72,
    aspect_weight: float = 0.28,
    south_shadow_boost: float = 1.35,
    compression: float = 0.88,
    min_snow: float = 0.12,
    shade_min: float = 0.55,
    sun_max: float = 1.35,
    protect_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Cartographic exposition shading — run last so summer grading cannot flatten it."""
    hill = compress_hillshade(hillshade_generalized, compression)
    hill_term = (hill - 0.5) * 2.0 * float(hillshade_weight)
    aspect_term = aspect_relief_for_sun(
        aspect,
        sun_azimuth,
        float(aspect_weight),
        shadow_boost=south_shadow_boost,
    )
    slope_factor = np.clip(slope / 8.0, 0.15, 1.0)
    relief = (hill_term + aspect_term * slope_factor) * float(strength)

    snow_weight = np.clip(
        (snow_fraction - min_snow) / max(1.0 - min_snow, 1e-3),
        0.0,
        1.0,
    )
    pixel_weight = snow_weight
    if protect_mask is not None:
        pixel_weight = pixel_weight * (1.0 - protect_mask.astype(np.float32))
    relief = relief * pixel_weight
    scale = np.clip(1.0 + relief, shade_min, sun_max)

    lum = luminance(rgb)
    scaled_lum = np.clip(lum * scale, 0.0, 1.0)
    lum_safe = np.maximum(lum, 1e-4)
    return np.clip(rgb * (scaled_lum / lum_safe)[..., np.newaxis], 0.0, 1.0)


def apply_winter_relief(
    rgb: np.ndarray,
    *,
    hillshade: np.ndarray,
    aspect: np.ndarray,
    snow_fraction: np.ndarray,
    snow_color: np.ndarray,
    summer_rgb: np.ndarray | None = None,
    hillshade_generalized: np.ndarray | None = None,
    cast_shadow: np.ndarray | None = None,
    relief_weight: np.ndarray | None = None,
    hillshade_strength: float = 0.12,
    aspect_strength: float = 0.06,
    compression: float = 0.35,
    macro_hillshade_weight: float = 0.82,
    min_snow: float = 0.30,
    shade_min: float = 0.82,
    sun_max: float = 1.18,
    snow_tint_strength: float = 0.50,
    summer_relief_weight: float = 0.0,
    cast_shadow_relief_weight: float = 0.0,
    sun_azimuth: float = 45.0,
    south_shadow_boost: float = 1.0,
) -> np.ndarray:
    """Subtle hillshade relief on snow; snowy pixels keep snow chroma, not neutral gray."""
    from winter_ortho.rendering.summer_structure import blend_macro_hillshade, summer_luminance_map

    snow_weight = np.clip(
        (snow_fraction - min_snow) / max(1.0 - min_snow, 1e-3),
        0.0,
        1.0,
    )
    weight = np.ones_like(snow_fraction, dtype=np.float32) if relief_weight is None else relief_weight
    weight = weight * snow_weight

    hill = blend_macro_hillshade(
        hillshade,
        hillshade_generalized,
        macro_weight=macro_hillshade_weight,
        compression=compression,
    )
    hill_relief = (hill - 0.5) * 2.0 * hillshade_strength
    aspect_relief = aspect_relief_for_sun(
        aspect,
        sun_azimuth,
        aspect_strength,
        shadow_boost=south_shadow_boost,
    )
    relief = hill_relief + aspect_relief
    if summer_rgb is not None and summer_relief_weight > 0:
        summer_map = summer_luminance_map(summer_rgb)
        summer_smooth = ndimage.gaussian_filter(
            summer_map.astype(np.float64),
            sigma=5.0,
        ).astype(np.float32)
        smooth_mix = np.clip(snow_weight * 0.90, 0.0, 0.90)
        summer_map = summer_map * (1.0 - smooth_mix) + summer_smooth * smooth_mix
        summer_field = summer_map - 0.5
        # Thick snow should read as a smooth deck, not inherit summer micro-texture.
        effective_summer = float(summer_relief_weight) * (1.0 - snow_weight * 0.88)
        relief = (
            relief * (1.0 - effective_summer)
            + summer_field * 2.0 * hillshade_strength * effective_summer
        )
    if cast_shadow is not None and cast_shadow_relief_weight > 0:
        relief = relief - cast_shadow * cast_shadow_relief_weight * hillshade_strength * 2.2
    relief = relief * weight
    scale = np.clip(1.0 + relief, shade_min, sun_max)

    lum = luminance(rgb)
    scaled_lum = np.clip(lum * scale, 0.0, 1.0)
    lum_safe = np.maximum(lum, 1e-4)
    hue_preserved = rgb * (scaled_lum / lum_safe)[..., np.newaxis]

    snow_lum = max(float(luminance(np.asarray(snow_color).reshape(1, 1, 3)).item()), 1e-4)
    snow_unit = snow_color / snow_lum
    snow_tinted = np.clip(scaled_lum[..., np.newaxis] * snow_unit, 0.0, 1.0)
    tint_strength = np.clip(weight * snow_tint_strength, 0.0, 1.0)
    return np.clip(
        hue_preserved * (1.0 - tint_strength[..., np.newaxis])
        + snow_tinted * tint_strength[..., np.newaxis],
        0.0,
        1.0,
    )
