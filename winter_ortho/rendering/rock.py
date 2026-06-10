from __future__ import annotations

import numpy as np

from winter_ortho.rendering.base import blend
from winter_ortho.rendering.relief import shade_snow_layer


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
) -> np.ndarray:
    out = rgb.copy()
    active = mask > 0
    if not active.any():
        return out

    snow_layer = shade_snow_layer(
        snow_color,
        hillshade,
        hillshade_strength,
        compression=hillshade_compression,
        snow_fraction=snow_fraction,
        snow_flattening=snow_flattening,
    )

    slope_t = np.clip(
        (slope - gentle_slope_max_deg) / max(steep_slope_min_deg - gentle_slope_max_deg, 1e-3),
        0,
        1,
    )
    gentle_factor = 1.0 - slope_t
    alpha = snow_fraction * (1.0 - rock_visibility) + gentle_factor * gentle_snow_boost
    alpha = np.clip(alpha, 0.0, 0.95)

    blended = blend(out, snow_layer, alpha[..., np.newaxis])
    out[active] = blended[active]
    return out
