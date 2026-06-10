from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import rasterio.features
from rasterio.transform import Affine

from winter_ortho.utils.raster import TargetGrid


def rasterize_layer(
    gdf: gpd.GeoDataFrame,
    grid: TargetGrid,
    *,
    burn_value: int = 1,
    all_touched: bool = True,
    buffer_m: float = 0.0,
) -> np.ndarray:
    if gdf.empty:
        return np.zeros((grid.height, grid.width), dtype=np.uint8)

    shapes = []
    geometries = gdf.geometry
    if buffer_m > 0:
        geometries = geometries.buffer(buffer_m)

    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        shapes.append((geom, burn_value))

    if not shapes:
        return np.zeros((grid.height, grid.width), dtype=np.uint8)

    return rasterio.features.rasterize(
        shapes=shapes,
        out_shape=(grid.height, grid.width),
        transform=grid.transform,
        fill=0,
        dtype=np.uint8,
        all_touched=all_touched,
    )


def rasterize_masks(
    mask_arrays: dict[str, np.ndarray],
    priority: list[str],
) -> np.ndarray:
    height, width = next(iter(mask_arrays.values())).shape
    combined = np.zeros((height, width), dtype=np.uint8)
    for name in priority:
        if name not in mask_arrays:
            continue
        layer = mask_arrays[name] > 0
        combined[layer] = mask_arrays[name][layer]
    return combined


def apply_priority(
    class_masks: dict[str, np.ndarray],
    priority: list[str],
    mask_values: dict[str, int],
) -> np.ndarray:
    height, width = next(iter(class_masks.values())).shape
    combined = np.zeros((height, width), dtype=np.uint8)
    for name in reversed(priority):
        mask = class_masks.get(name)
        if mask is None:
            continue
        value = mask_values.get(name, 0)
        combined[mask > 0] = value
    return combined
