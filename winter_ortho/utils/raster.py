from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject


@dataclass(frozen=True)
class TargetGrid:
    crs: str
    transform: Affine
    width: int
    height: int
    bounds: tuple[float, float, float, float]

    @classmethod
    def from_bbox(
        cls,
        bbox: tuple[float, float, float, float],
        resolution_m: float,
        crs: str,
    ) -> TargetGrid:
        minx, miny, maxx, maxy = bbox
        width = int(round((maxx - minx) / resolution_m))
        height = int(round((maxy - miny) / resolution_m))
        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        return cls(
            crs=crs,
            transform=transform,
            width=width,
            height=height,
            bounds=bbox,
        )


def read_raster(path: str) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        data = src.read()
        meta = {
            "crs": src.crs.to_string() if src.crs else None,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "bounds": src.bounds,
        }
    return data, meta


def write_cog(
    path: str,
    data: np.ndarray,
    *,
    transform: Affine,
    crs: str,
    nodata: float | int | None = None,
    compress: str = "deflate",
    alpha_mask: np.ndarray | None = None,
    photometric: str | None = None,
) -> None:
    if data.ndim == 2:
        count = 1
        height, width = data.shape
    else:
        count, height, width = data.shape

    if alpha_mask is not None:
        if alpha_mask.shape != (height, width):
            raise ValueError(
                f"alpha_mask shape {alpha_mask.shape} does not match raster {(height, width)}"
            )
        alpha = np.where(alpha_mask, 255, 0).astype(np.uint8)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        data = np.concatenate([data, alpha[np.newaxis, ...]], axis=0)
        count = data.shape[0]

    if photometric is None and count >= 3 and data.dtype == np.uint8:
        photometric = "RGBA" if alpha_mask is not None else "RGB"

    profile = {
        "driver": "GTiff",
        "dtype": data.dtype,
        "width": width,
        "height": height,
        "count": count,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": compress,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "interleave": "pixel",
    }
    if photometric is not None:
        profile["photometric"] = photometric

    with rasterio.open(path, "w", **profile) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            dst.write(data)
        if count >= 3 and photometric in ("RGB", "RGBA"):
            dst.colorinterp = (
                rasterio.enums.ColorInterp.red,
                rasterio.enums.ColorInterp.green,
                rasterio.enums.ColorInterp.blue,
            ) + (
                (rasterio.enums.ColorInterp.alpha,) if count >= 4 else ()
            )
        if alpha_mask is not None and count >= 4:
            dst.set_band_description(count, "Alpha")
            dst.colorinterp = (
                rasterio.enums.ColorInterp.red,
                rasterio.enums.ColorInterp.green,
                rasterio.enums.ColorInterp.blue,
                rasterio.enums.ColorInterp.alpha,
            )[:count]


def reproject_to_grid(
    source_path: str,
    grid: TargetGrid,
    *,
    resampling: Resampling,
    band_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(source_path) as src:
        count = band_count or src.count
        destination = np.zeros((count, grid.height, grid.width), dtype=np.float32)
        nodata_mask = np.ones((grid.height, grid.width), dtype=bool)

        for band in range(1, count + 1):
            band_data = np.zeros((grid.height, grid.width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, band),
                destination=band_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                resampling=resampling,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
            destination[band - 1] = band_data
            valid = np.isfinite(band_data)
            if src.nodata is not None:
                valid &= band_data != src.nodata
            nodata_mask &= valid

        if count == 1:
            return destination[0], nodata_mask
        return destination, nodata_mask


def alignment_report(
    reference_meta: dict[str, Any],
    other_meta: dict[str, Any],
    *,
    resolution_tol: float = 1e-3,
    origin_tol: float = 0.5,
) -> dict[str, Any]:
    ref_res = (reference_meta["transform"].a, abs(reference_meta["transform"].e))
    other_res = (other_meta["transform"].a, abs(other_meta["transform"].e))
    origin_delta = (
        abs(reference_meta["transform"].c - other_meta["transform"].c),
        abs(reference_meta["transform"].f - other_meta["transform"].f),
    )
    resolution_match = (
        abs(ref_res[0] - other_res[0]) < resolution_tol
        and abs(ref_res[1] - other_res[1]) < resolution_tol
    )
    origin_match = origin_delta[0] < origin_tol and origin_delta[1] < origin_tol
    return {
        "resolution_match": resolution_match,
        "origin_match": origin_match,
        "reference_resolution": ref_res,
        "other_resolution": other_res,
        "origin_delta_m": origin_delta,
        "size_match": (
            reference_meta["width"] == other_meta["width"]
            and reference_meta["height"] == other_meta["height"]
        ),
        "aligned": resolution_match and origin_match and (
            reference_meta["width"] == other_meta["width"]
            and reference_meta["height"] == other_meta["height"]
        ),
    }
