from __future__ import annotations

import numpy as np

from winter_ortho.rendering.base import blend


def render_roads(
    rgb: np.ndarray,
    road_mask: np.ndarray,
    *,
    snow_fraction: np.ndarray,
    road_visibility: np.ndarray,
    road_color: np.ndarray,
) -> np.ndarray:
    out = rgb.copy()
    active = road_mask > 0
    if not active.any():
        return out

    alpha = snow_fraction * road_visibility
    blended = blend(out, road_color, alpha[..., np.newaxis])
    out[active] = blended[active]
    return out
