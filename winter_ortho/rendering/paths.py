from __future__ import annotations

import numpy as np

from winter_ortho.features.texture import band_limited_noise
from winter_ortho.rendering.base import blend, modulate_snow_layer_brightness, snow_cover_alpha
from winter_ortho.rendering.relief import shade_snow_layer
from winter_ortho.rendering.summer_structure import combined_shade_field


def render_paths(
    rgb: np.ndarray,
    path_mask: np.ndarray,
    *,
    snow_fraction: np.ndarray,
    snow_brightness: np.ndarray,
    snow_texture_strength: np.ndarray,
    snow_color: np.ndarray,
    hillshade: np.ndarray,
    bury_strength: float,
    max_snow_blend: float = 0.82,
    hillshade_strength: float,
    hillshade_compression: float = 0.35,
    snow_flattening: float = 0.75,
    noise_scale_px: int,
) -> np.ndarray:
    """Fully bury hiking paths under snow — path geometry should not remain visible."""
    out = rgb.copy()
    active = path_mask > 0
    if not active.any():
        return out

    summer = out.copy()
    shade_field = combined_shade_field(
        hillshade,
        summer,
        active,
        hillshade_weight=0.40,
        summer_weight=0.60,
        compression=hillshade_compression,
    )
    noise = band_limited_noise(rgb.shape[:2], scale_px=noise_scale_px, seed=31)
    snow_layer = shade_snow_layer(
        snow_color,
        hillshade,
        hillshade_strength,
        compression=hillshade_compression,
        snow_fraction=snow_fraction,
        snow_flattening=snow_flattening,
        shade_field=shade_field,
        shadow_boost=1.2,
        highlight_cap=0.48,
    )
    texture = noise[..., np.newaxis] * snow_texture_strength[..., np.newaxis] * 0.1
    snow_layer = np.clip(snow_layer + texture, 0, 1)
    snow_layer = modulate_snow_layer_brightness(snow_layer, snow_brightness, snow_color)

    alpha = snow_cover_alpha(
        snow_fraction,
        max_snow_blend=max_snow_blend,
        strength=bury_strength,
    )
    blended = blend(out, snow_layer, alpha)
    out[active] = blended[active]
    return out
