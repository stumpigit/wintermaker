from __future__ import annotations

import numpy as np

from winter_ortho.rendering.base import blend
from winter_ortho.rendering.relief import desaturate, shade_snow_layer
from winter_ortho.rendering.summer_structure import combined_shade_field


def render_rock(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    snow_fraction: np.ndarray,
    rock_visibility: np.ndarray,
    slope: np.ndarray,
    snow_color: np.ndarray,
    hillshade: np.ndarray,
    hillshade_strength: float,
    hillshade_compression: float = 0.55,
    snow_flattening: float = 0.35,
    gentle_slope_max_deg: float = 28.0,
    steep_slope_min_deg: float = 42.0,
    gentle_snow_boost: float = 0.2,
    summer_shade_weight: float = 0.55,
    hillshade_shade_weight: float = 0.45,
    shadow_boost: float = 1.45,
    highlight_cap: float = 0.50,
    summer_preservation: float = 0.42,
    **_: object,
) -> np.ndarray:
    out = rgb.copy()
    active = mask > 0
    if not active.any():
        return out

    summer = out.copy()
    summer_rock = desaturate(summer, np.where(active, 0.45, 0.0))
    shade_field = combined_shade_field(
        hillshade,
        summer,
        active,
        hillshade_weight=hillshade_shade_weight,
        summer_weight=summer_shade_weight,
        compression=hillshade_compression,
    )
    snow_layer = shade_snow_layer(
        snow_color,
        hillshade,
        hillshade_strength,
        compression=hillshade_compression,
        snow_fraction=snow_fraction,
        snow_flattening=snow_flattening,
        shade_field=shade_field,
        shadow_boost=shadow_boost,
        highlight_cap=highlight_cap,
    )

    slope_t = np.clip(
        (slope - gentle_slope_max_deg) / max(steep_slope_min_deg - gentle_slope_max_deg, 1e-3),
        0,
        1,
    )
    gentle_factor = 1.0 - slope_t
    alpha = snow_fraction * (1.0 - rock_visibility) + gentle_factor * gentle_snow_boost
    alpha = np.clip(alpha, 0.0, 0.96)

    snowy = blend(summer_rock, snow_layer, alpha[..., np.newaxis])
    preserve = np.clip(
        summer_preservation * (rock_visibility + 0.15) * (0.20 + 0.80 * slope_t),
        0.0,
        0.55,
    )
    blended = blend(snowy, summer_rock, preserve[..., np.newaxis])
    out[active] = blended[active]
    return out
