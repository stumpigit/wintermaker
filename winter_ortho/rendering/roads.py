from __future__ import annotations

import numpy as np

from winter_ortho.rendering.base import blend
from winter_ortho.rendering.relief import desaturate, luminance


def render_roads(
    rgb: np.ndarray,
    road_mask: np.ndarray,
    summer_rgb: np.ndarray,
    *,
    snow_fraction: np.ndarray,
    road_visibility: np.ndarray,
    road_color: np.ndarray,
    summer_line_strength: float = 0.72,
    min_visibility: float = 0.38,
) -> np.ndarray:
    out = rgb.copy()
    active = road_mask > 0
    if not active.any():
        return out

    summer_lum = luminance(summer_rgb)
    summer_dark = np.clip((0.58 - summer_lum) / 0.38, 0.0, 1.0)
    summer_road = desaturate(summer_rgb, 0.65)
    road_pixels = blend(summer_road, road_color, 0.45)

    alpha = np.clip(
        road_visibility * (min_visibility + (1.0 - min_visibility) * summer_dark * summer_line_strength),
        0.20,
        0.90,
    )
    alpha = alpha * (1.0 - snow_fraction * 0.25)
    blended = blend(out, road_pixels, alpha[..., np.newaxis])
    out[active] = blended[active]
    return out
