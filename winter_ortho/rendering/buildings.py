from __future__ import annotations

import numpy as np
from scipy import ndimage

from winter_ortho.rendering.base import blend
from winter_ortho.rendering.relief import desaturate, luminance
from winter_ortho.rendering.summer_structure import summer_luminance_map


def render_buildings(
    rgb: np.ndarray,
    mask: np.ndarray,
    summer_rgb: np.ndarray,
    *,
    roof_snow_intensity: np.ndarray,
    snow_color: np.ndarray,
    brighten_factor: float,
    edge_preservation: float,
    wall_preservation: float = 0.62,
) -> np.ndarray:
    out = rgb.copy()
    active = mask > 0
    if not active.any():
        return out

    edges = ndimage.sobel(mask.astype(np.float32), axis=0) ** 2
    edges += ndimage.sobel(mask.astype(np.float32), axis=1) ** 2
    edges = edges > 0

    summer_building = desaturate(summer_rgb, 0.55)
    roof_signal = summer_luminance_map(summer_rgb, active)
    alpha = np.clip(roof_snow_intensity * brighten_factor * (0.25 + 0.75 * roof_signal), 0.0, 0.75)

    wall_alpha = np.clip(wall_preservation * (1.0 - alpha * 0.6), 0.0, 0.85)
    winter = blend(out, summer_building, wall_alpha[..., np.newaxis])
    winter = blend(winter, snow_color, alpha[..., np.newaxis])

    out[active] = winter[active]
    if edge_preservation > 0:
        edge_blend = summer_rgb * edge_preservation + out * (1.0 - edge_preservation)
        out[edges] = edge_blend[edges]
    return out
