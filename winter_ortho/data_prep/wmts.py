from __future__ import annotations

import math
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

WMTS_CAPABILITIES_2056 = (
    "https://wmts.geo.admin.ch/EPSG/2056/1.0.0/WMTSCapabilities.xml"
)
SWISSIMAGE_LAYER = "ch.swisstopo.swissimage"
SWISSIMAGE_MATRIX_SET = "2056_28"
PIXEL_SIZE_FACTOR = 0.00028
WMTS_NS = {
    "w": "http://www.opengis.net/wmts/1.0",
    "o": "http://www.opengis.net/ows/1.1",
}


@dataclass(frozen=True)
class TileMatrix:
    identifier: str
    scale_denominator: float
    top_left_x: float
    top_left_y: float
    tile_width: int
    tile_height: int

    @property
    def tile_size_m(self) -> float:
        return self.tile_width * self.scale_denominator * PIXEL_SIZE_FACTOR

    @property
    def resolution_m(self) -> float:
        return self.scale_denominator * PIXEL_SIZE_FACTOR


@dataclass(frozen=True)
class TileRange:
    zoom: int
    col_min: int
    col_max: int
    row_min: int
    row_max: int
    matrix: TileMatrix


def parse_extent(extent: str) -> tuple[float, float, float, float]:
    parts = [float(v.strip()) for v in extent.split(",")]
    if len(parts) != 4:
        raise ValueError("extent must be minx,miny,maxx,maxy in EPSG:2056")
    minx, miny, maxx, maxy = parts
    if maxx <= minx or maxy <= miny:
        raise ValueError(f"Invalid extent: {extent}")
    return minx, miny, maxx, maxy


@lru_cache(maxsize=4)
def _load_capabilities(url: str) -> ET.Element:
    with urllib.request.urlopen(url, timeout=60) as response:
        return ET.fromstring(response.read())


def _parse_tile_matrix(element: ET.Element) -> TileMatrix:
    top_left = element.find("w:TopLeftCorner", WMTS_NS).text.split()
    return TileMatrix(
        identifier=element.find("o:Identifier", WMTS_NS).text,
        scale_denominator=float(element.find("w:ScaleDenominator", WMTS_NS).text),
        top_left_x=float(top_left[0]),
        top_left_y=float(top_left[1]),
        tile_width=int(element.find("w:TileWidth", WMTS_NS).text),
        tile_height=int(element.find("w:TileHeight", WMTS_NS).text),
    )


def list_tile_matrices(
    matrix_set: str = SWISSIMAGE_MATRIX_SET,
    *,
    capabilities_url: str = WMTS_CAPABILITIES_2056,
) -> list[TileMatrix]:
    root = _load_capabilities(capabilities_url)
    for tile_matrix_set in root.findall(".//w:TileMatrixSet", WMTS_NS):
        identifier = tile_matrix_set.find("o:Identifier", WMTS_NS)
        if identifier is None or identifier.text != matrix_set:
            continue
        matrices = tile_matrix_set.findall("w:TileMatrix", WMTS_NS)
        if not matrices:
            raise ValueError(f"No tile matrices in {matrix_set}")
        return [_parse_tile_matrix(matrix) for matrix in matrices]
    raise ValueError(f"Tile matrix set not found: {matrix_set}")


def get_tile_matrix(
    matrix_set: str = SWISSIMAGE_MATRIX_SET,
    zoom: int | None = None,
    *,
    capabilities_url: str = WMTS_CAPABILITIES_2056,
) -> TileMatrix:
    matrices = list_tile_matrices(matrix_set, capabilities_url=capabilities_url)
    if zoom is None:
        return matrices[-1]
    for matrix in matrices:
        if matrix.identifier == str(zoom):
            return matrix
    raise ValueError(f"Zoom level {zoom} not found in {matrix_set}")


def zoom_for_resolution(
    resolution_m: float,
    *,
    matrix_set: str = SWISSIMAGE_MATRIX_SET,
) -> int:
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be positive, got {resolution_m}")
    matrices = list_tile_matrices(matrix_set)
    best = min(
        matrices,
        key=lambda matrix: (abs(matrix.resolution_m - resolution_m), -int(matrix.identifier)),
    )
    return int(best.identifier)


def _tile_index_range(
    value_min: float,
    value_max: float,
    origin: float,
    tile_size: float,
    *,
    descending: bool = False,
) -> tuple[int, int]:
    if descending:
        index_min = math.floor((origin - value_max) / tile_size)
        index_max = math.floor((origin - value_min) / tile_size)
    else:
        index_min = math.floor((value_min - origin) / tile_size)
        index_max = math.floor((value_max - origin) / tile_size)
    return index_min, index_max


def tile_range_for_bbox(
    bbox: tuple[float, float, float, float],
    *,
    zoom: int | None = None,
) -> TileRange:
    matrix = get_tile_matrix(zoom=zoom)
    minx, miny, maxx, maxy = bbox
    tile_size = matrix.tile_size_m
    col_min, col_max = _tile_index_range(minx, maxx, matrix.top_left_x, tile_size)
    row_min, row_max = _tile_index_range(
        miny, maxy, matrix.top_left_y, tile_size, descending=True
    )
    return TileRange(
        zoom=int(matrix.identifier),
        col_min=col_min,
        col_max=col_max,
        row_min=row_min,
        row_max=row_max,
        matrix=matrix,
    )


def tile_bounds(col: int, row: int, matrix: TileMatrix) -> tuple[float, float, float, float]:
    tile_size = matrix.tile_size_m
    minx = matrix.top_left_x + col * tile_size
    maxy = matrix.top_left_y - row * tile_size
    maxx = minx + tile_size
    miny = maxy - tile_size
    return minx, miny, maxx, maxy


def swissimage_tile_url(col: int, row: int, zoom: int) -> str:
    return (
        "https://wmts.geo.admin.ch/1.0.0/"
        f"{SWISSIMAGE_LAYER}/default/current/2056/{zoom}/{col}/{row}.jpeg"
    )


def describe_tile_range(tile_range: TileRange) -> dict[str, object]:
    cols = range(tile_range.col_min, tile_range.col_max + 1)
    rows = range(tile_range.row_min, tile_range.row_max + 1)
    return {
        "zoom": tile_range.zoom,
        "col_range": [tile_range.col_min, tile_range.col_max],
        "row_range": [tile_range.row_min, tile_range.row_max],
        "tile_count": len(cols) * len(rows),
        "native_resolution_m": tile_range.matrix.resolution_m,
        "tile_size_m": tile_range.matrix.tile_size_m,
        "sample_url": swissimage_tile_url(tile_range.col_min, tile_range.row_min, tile_range.zoom),
    }


def fetch_swissimage_mosaic(
    bbox: tuple[float, float, float, float],
    output_path: str,
    *,
    resolution_m: float,
    zoom: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    if zoom is None:
        zoom = zoom_for_resolution(resolution_m)
    tile_range = tile_range_for_bbox(bbox, zoom=zoom)
    matrix = tile_range.matrix
    cols = range(tile_range.col_min, tile_range.col_max + 1)
    rows = range(tile_range.row_min, tile_range.row_max + 1)
    total = len(cols) * len(rows)

    mosaic_minx, _, _, mosaic_maxy = tile_bounds(
        tile_range.col_min, tile_range.row_min, matrix
    )
    _, mosaic_miny, mosaic_maxx, _ = tile_bounds(
        tile_range.col_max, tile_range.row_max, matrix
    )

    tile_size_px = matrix.tile_width
    tile_size_m = matrix.tile_size_m
    width_px = int(round((mosaic_maxx - mosaic_minx) / tile_size_m * tile_size_px))
    height_px = int(round((mosaic_maxy - mosaic_miny) / tile_size_m * tile_size_px))
    mosaic = np.zeros((height_px, width_px, 3), dtype=np.uint8)

    downloaded = 0
    for row in rows:
        for col in cols:
            url = swissimage_tile_url(col, row, tile_range.zoom)
            if progress:
                progress(f"WMTS {downloaded + 1}/{total}: {col}/{row}")
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = response.read()
            except urllib.error.HTTPError as exc:
                raise RuntimeError(f"Failed to download tile {url}: {exc}") from exc

            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to decode tile {url}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            tile_minx, tile_miny, _, _ = tile_bounds(col, row, matrix)
            x0 = int(round((tile_minx - mosaic_minx) / tile_size_m * tile_size_px))
            y0 = int(round((mosaic_maxy - (tile_miny + tile_size_m)) / tile_size_m * tile_size_px))
            mosaic[y0 : y0 + tile_size_px, x0 : x0 + tile_size_px] = image
            downloaded += 1

    minx, miny, maxx, maxy = bbox
    x_off = int(round((minx - mosaic_minx) / tile_size_m * tile_size_px))
    y_off = int(round((mosaic_maxy - maxy) / tile_size_m * tile_size_px))
    crop_w = int(round((maxx - minx) / tile_size_m * tile_size_px))
    crop_h = int(round((maxy - miny) / tile_size_m * tile_size_px))
    cropped = mosaic[y_off : y_off + crop_h, x_off : x_off + crop_w]

    native_transform = from_bounds(minx, miny, maxx, maxy, crop_w, crop_h)
    out_w = int(round((maxx - minx) / resolution_m))
    out_h = int(round((maxy - miny) / resolution_m))
    out_transform = from_bounds(minx, miny, maxx, maxy, out_w, out_h)

    resampled = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    for band in range(3):
        reproject(
            source=cropped[:, :, band],
            destination=resampled[:, :, band],
            src_transform=native_transform,
            src_crs="EPSG:2056",
            dst_transform=out_transform,
            dst_crs="EPSG:2056",
            resampling=Resampling.bilinear,
        )

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=out_h,
        width=out_w,
        count=3,
        dtype="uint8",
        crs="EPSG:2056",
        transform=out_transform,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        photometric="RGB",
    ) as dataset:
        for band in range(3):
            dataset.write(resampled[:, :, band], band + 1)

    tile_info = describe_tile_range(tile_range)
    return {
        **tile_info,
        "target_resolution_m": resolution_m,
        "width": out_w,
        "height": out_h,
        "output": output_path,
    }
