from __future__ import annotations

import json
import math
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds

SWISSALTI3D_BASE = "https://data.geo.admin.ch/ch.swisstopo.swissalti3d"
STAC_ITEMS_URL = (
    "https://data.geo.admin.ch/api/stac/v1/collections/ch.swisstopo.swissalti3d/items"
)
DEFAULT_DEM_YEAR = 2023
DEM_ASSET_SUFFIX = "_2_2056_5728.tif"
CANDIDATE_YEARS = list(range(2025, 2018, -1))


@dataclass(frozen=True)
class DemTile:
    easting_km: int
    northing_km: int
    year: int
    url: str

    @property
    def cell_id(self) -> str:
        return f"{self.easting_km}-{self.northing_km}"


def dem_tile_indices(
    bbox: tuple[float, float, float, float],
) -> tuple[range, range]:
    minx, miny, maxx, maxy = bbox
    east_min = int(math.floor(minx / 1000.0))
    east_max = int(math.floor((maxx - 1e-6) / 1000.0))
    north_min = int(math.floor(miny / 1000.0))
    north_max = int(math.floor((maxy - 1e-6) / 1000.0))
    if east_max < east_min or north_max < north_min:
        raise ValueError(f"No DEM tiles for bbox: {bbox}")
    return range(east_min, east_max + 1), range(north_min, north_max + 1)


def swissalti3d_tile_url(easting_km: int, northing_km: int, year: int = DEFAULT_DEM_YEAR) -> str:
    stem = f"swissalti3d_{year}_{easting_km}-{northing_km}"
    filename = f"{stem}{DEM_ASSET_SUFFIX}"
    return f"{SWISSALTI3D_BASE}/{stem}/{filename}"


def _candidate_years(preferred_year: int | None) -> list[int]:
    if preferred_year is None:
        return CANDIDATE_YEARS
    years = [preferred_year]
    for year in CANDIDATE_YEARS:
        if year not in years:
            years.append(year)
    return years


def _tile_url_exists(url: str) -> bool:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return 200 <= response.status < 300
    except urllib.error.HTTPError:
        return False


def _parse_stac_feature(feature: dict) -> tuple[str, int, str] | None:
    feature_id = feature["id"]
    parts = feature_id.split("_")
    if len(parts) < 3:
        return None
    try:
        year = int(parts[1])
    except ValueError:
        return None
    cell_id = parts[2]
    asset_key = f"{feature_id}{DEM_ASSET_SUFFIX}"
    asset = feature.get("assets", {}).get(asset_key)
    if asset is None:
        return None
    return cell_id, year, asset["href"]


def query_stac_dem_tiles(bbox: tuple[float, float, float, float]) -> dict[str, tuple[int, str]]:
    minx, miny, maxx, maxy = bbox
    wgs_bbox = transform_bounds("EPSG:2056", "EPSG:4326", minx, miny, maxx, maxy)
    query = (
        f"{STAC_ITEMS_URL}?bbox={wgs_bbox[0]},{wgs_bbox[1]},{wgs_bbox[2]},{wgs_bbox[3]}"
        "&limit=500"
    )
    with urllib.request.urlopen(query, timeout=60) as response:
        payload = json.load(response)

    best: dict[str, tuple[int, str]] = {}
    for feature in payload.get("features", []):
        parsed = _parse_stac_feature(feature)
        if parsed is None:
            continue
        cell_id, year, url = parsed
        if cell_id not in best or year > best[cell_id][0]:
            best[cell_id] = (year, url)
    return best


def resolve_dem_tiles(
    bbox: tuple[float, float, float, float],
    *,
    preferred_year: int | None = DEFAULT_DEM_YEAR,
) -> list[DemTile]:
    eastings, northings = dem_tile_indices(bbox)
    stac_tiles = query_stac_dem_tiles(bbox)
    resolved: list[DemTile] = []
    missing: list[str] = []

    for easting in eastings:
        for northing in northings:
            cell_id = f"{easting}-{northing}"
            stac_match = stac_tiles.get(cell_id)
            if stac_match is not None:
                year, url = stac_match
                if preferred_year is not None and _tile_url_exists(
                    swissalti3d_tile_url(easting, northing, preferred_year)
                ):
                    year = preferred_year
                    url = swissalti3d_tile_url(easting, northing, preferred_year)
                resolved.append(DemTile(easting, northing, year, url))
                continue

            found: DemTile | None = None
            for year in _candidate_years(preferred_year):
                url = swissalti3d_tile_url(easting, northing, year)
                if _tile_url_exists(url):
                    found = DemTile(easting, northing, year, url)
                    break
            if found is None:
                missing.append(cell_id)
            else:
                resolved.append(found)

    if missing:
        raise RuntimeError(
            "No swissALTI3D 2 m tile found for cell(s): "
            + ", ".join(missing)
        )
    return resolved


def iter_dem_tile_urls(
    bbox: tuple[float, float, float, float],
    *,
    year: int = DEFAULT_DEM_YEAR,
) -> Iterable[tuple[int, int, str]]:
    for tile in resolve_dem_tiles(bbox, preferred_year=year):
        yield tile.easting_km, tile.northing_km, tile.url


def _download_tile(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            destination.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Failed to download DEM tile {url}: {exc}") from exc


def fetch_swissalti3d_mosaic(
    bbox: tuple[float, float, float, float],
    output_path: str | Path,
    *,
    year: int = DEFAULT_DEM_YEAR,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tiles = resolve_dem_tiles(bbox, preferred_year=year)
    if not tiles:
        raise ValueError(f"No DEM tiles for bbox: {bbox}")

    if progress:
        years = sorted({tile.year for tile in tiles})
        progress(f"DEM: {len(tiles)} Kacheln, Jahrgänge {years}")

    with tempfile.TemporaryDirectory(prefix="swissalti3d_") as temp_dir:
        temp_root = Path(temp_dir)
        datasets = []
        for index, tile in enumerate(tiles, start=1):
            local_path = temp_root / f"{tile.easting_km}_{tile.northing_km}.tif"
            if progress:
                progress(
                    f"DEM {index}/{len(tiles)}: {tile.cell_id} "
                    f"({tile.year})"
                )
            _download_tile(tile.url, local_path)
            datasets.append(rasterio.open(local_path))

        mosaic, transform = merge(datasets)
        for dataset in datasets:
            dataset.close()

    data = mosaic[0]
    nodata = -9999.0
    minx, miny, maxx, maxy = bbox
    width = int(round((maxx - minx) / 2.0))
    height = int(round((maxy - miny) / 2.0))
    target_transform = from_bounds(minx, miny, maxx, maxy, width, height)

    cropped = np.full((height, width), nodata, dtype=np.float32)
    reproject(
        source=data,
        destination=cropped,
        src_transform=transform,
        src_crs="EPSG:2056",
        dst_transform=target_transform,
        dst_crs="EPSG:2056",
        resampling=Resampling.bilinear,
        src_nodata=nodata,
        dst_nodata=nodata,
    )

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:2056",
        transform=target_transform,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dataset:
        dataset.write(cropped, 1)

    valid = cropped[cropped != nodata]
    return {
        "tile_count": len(tiles),
        "tile_years": sorted({tile.year for tile in tiles}),
        "width": width,
        "height": height,
        "output": str(output_path),
        "elevation_min": float(valid.min()) if valid.size else None,
        "elevation_max": float(valid.max()) if valid.size else None,
        "elevation_mean": float(valid.mean()) if valid.size else None,
    }
