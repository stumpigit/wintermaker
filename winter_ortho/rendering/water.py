from __future__ import annotations

import numpy as np


def render_water(
    rgb: np.ndarray,
    water_mask: np.ndarray,
    *,
    darken: float = 0.85,
) -> np.ndarray:
    out = rgb.copy()
    mask = water_mask > 0
    out[mask] = out[mask] * darken
    return out
