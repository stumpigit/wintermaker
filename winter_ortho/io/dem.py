from __future__ import annotations

from pathlib import Path

import numpy as np
from rasterio.enums import Resampling

from winter_ortho.utils.raster import TargetGrid, reproject_to_grid


def load_and_align_dem(
    source_path: str | Path,
    grid: TargetGrid,
    *,
    resampling: str = "bilinear",
) -> tuple[np.ndarray, np.ndarray]:
    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }
    data, valid_mask = reproject_to_grid(
        str(source_path),
        grid,
        resampling=resampling_map.get(resampling, Resampling.bilinear),
        band_count=1,
    )
    return data.astype(np.float32), valid_mask
