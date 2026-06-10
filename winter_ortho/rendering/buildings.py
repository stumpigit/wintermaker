from __future__ import annotations

import numpy as np
from scipy import ndimage

from winter_ortho.rendering.base import blend


def render_buildings(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    roof_snow_intensity: np.ndarray,
    snow_color: np.ndarray,
    brighten_factor: float,
    edge_preservation: float,
) -> np.ndarray:
    out = rgb.copy()
    active = mask > 0
    if not active.any():
        return out

    edges = ndimage.sobel(mask.astype(np.float32), axis=0) ** 2
    edges += ndimage.sobel(mask.astype(np.float32), axis=1) ** 2
    edges = edges > 0

    alpha = roof_snow_intensity * brighten_factor
    blended = blend(out, snow_color, alpha)
    out[active] = blended[active]
    if edge_preservation > 0:
        out[edges] = rgb[edges] * edge_preservation + out[edges] * (1 - edge_preservation)
    return out
