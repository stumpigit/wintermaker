from __future__ import annotations

import numpy as np

from winter_ortho.features.texture import band_limited_noise
from winter_ortho.rendering.base import blend, modulate_snow_layer_brightness, snow_cover_alpha
from winter_ortho.rendering.relief import shade_snow_layer


def render_settlement(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    snow_fraction: np.ndarray,
    snow_brightness: np.ndarray,
    snow_texture_strength: np.ndarray,
    snow_color: np.ndarray,
    hillshade: np.ndarray,
    hillshade_strength: float,
    hillshade_compression: float = 0.40,
    snow_flattening: float = 0.55,
    original_texture_visibility: float = 0.25,
    max_snow_blend: float = 0.70,
    noise_scale_px: int,
) -> np.ndarray:
    out = rgb.copy()
    active = mask > 0
    if not active.any():
        return out

    noise = band_limited_noise(rgb.shape[:2], scale_px=noise_scale_px, seed=23)
    snow_layer = shade_snow_layer(
        snow_color,
        hillshade,
        hillshade_strength,
        compression=hillshade_compression,
        snow_fraction=snow_fraction,
        snow_flattening=snow_flattening,
    )
    texture = noise[..., np.newaxis] * snow_texture_strength[..., np.newaxis] * 0.06
    snow_layer = np.clip(snow_layer + texture, 0, 1)
    snow_layer = modulate_snow_layer_brightness(snow_layer, snow_brightness, snow_color)

    summer = out.copy()
    alpha = snow_cover_alpha(snow_fraction, max_snow_blend=max_snow_blend)
    blended = blend(summer, snow_layer, alpha)
    texture_vis = original_texture_visibility * (1.0 - alpha)
    blended = blend(blended, summer, texture_vis)
    out[active] = blended[active]
    return out
