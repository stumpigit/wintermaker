from __future__ import annotations

import numpy as np
from scipy import ndimage


def band_limited_noise(
    shape: tuple[int, int],
    *,
    scale_px: int = 8,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, shape).astype(np.float32)
    sigma = max(scale_px / 2.0, 0.5)
    return ndimage.gaussian_filter(noise, sigma=sigma)
