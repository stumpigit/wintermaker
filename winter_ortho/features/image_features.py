from __future__ import annotations

import numpy as np
from skimage import color


def rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 3:
        rgb = np.moveaxis(rgb, 0, -1)
    lab = color.rgb2lab(rgb.astype(np.float32) / 255.0)
    return lab[..., 0] / 100.0
