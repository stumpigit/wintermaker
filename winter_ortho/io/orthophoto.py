from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from winter_ortho.utils.raster import TargetGrid, reproject_to_grid


def load_and_align_orthophoto(
    source_path: str | Path,
    grid: TargetGrid,
    *,
    resampling: str = "bilinear",
    max_bands: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }
    resampling_enum = resampling_map.get(resampling, Resampling.bilinear)

    with rasterio.open(str(source_path)) as src:
        has_alpha = src.count >= 4

    band_count = 4 if has_alpha else max_bands
    data, valid_mask = reproject_to_grid(
        str(source_path),
        grid,
        resampling=resampling_enum,
        band_count=band_count,
    )
    if data.ndim == 2:
        data = data[np.newaxis, ...]

    if has_alpha:
        alpha = data[3]
        valid_mask &= alpha >= 128

    rgb = data[:3]
    valid_mask &= rgb.sum(axis=0) > 12
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb, valid_mask
