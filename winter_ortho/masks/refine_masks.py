from __future__ import annotations

import numpy as np
from scipy import ndimage


def soften_mask_edges(mask: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    blurred = ndimage.gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return np.clip(blurred, 0.0, 1.0)
