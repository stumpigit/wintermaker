from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from winter_ortho.qa.visual_metrics import class_histogram
from winter_ortho.utils.raster import read_raster


def compare_to_reference(
    winter_path: str | Path,
    reference_path: str | Path,
    landcover_mask: np.ndarray,
) -> dict[str, Any]:
    if not Path(reference_path).exists():
        return {"available": False}

    winter, _ = read_raster(str(winter_path))
    reference, _ = read_raster(str(reference_path))
    if winter.ndim == 3:
        winter = np.moveaxis(winter[:3], 0, -1)
    if reference.ndim == 3:
        reference = np.moveaxis(reference[:3], 0, -1)

    return {
        "available": True,
        "winter_histogram": class_histogram(winter, landcover_mask > 0),
        "reference_histogram": class_histogram(reference, landcover_mask > 0),
    }
