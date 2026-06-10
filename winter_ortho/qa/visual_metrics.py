from __future__ import annotations

import numpy as np


def class_histogram(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if not mask.any():
        return {"mean_r": 0.0, "mean_g": 0.0, "mean_b": 0.0}
    pixels = rgb[mask > 0]
    return {
        "mean_r": float(pixels[:, 0].mean()),
        "mean_g": float(pixels[:, 1].mean()),
        "mean_b": float(pixels[:, 2].mean()),
    }
