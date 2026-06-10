from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from pyproj import Transformer

from winter_ortho.utils.config import load_config
from winter_ortho.utils.paths import get_project_root, tile_paths

NODATA = -9999.0
GPX_NS = "http://www.topografix.com/GPX/1/1"
TRACK_LIFT_M = 8.0
# Keep mesh coarse for browser memory; texture stays sharper separately.
MAX_GRID_DIM = 256
MAX_TEXTURE_DIM = 1024


def _update_manifest(data_root: Path, tile_id: str) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    manifest_path = data_root / "manifest.json"
    tiles: list[str] = []
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        tiles = list(manifest.get("tiles", []))
    if tile_id not in tiles:
        tiles.append(tile_id)
    tiles.sort()
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump({"tiles": tiles}, handle, indent=2)


def _auto_stride(width: int, height: int, max_dim: int = MAX_GRID_DIM) -> int:
    longest = max(width, height)
    if longest <= max_dim:
        return 1
    return int(np.ceil(longest / max_dim))


def _build_mesh(
    dem: np.ndarray,
    *,
    transform: rasterio.Affine,
    stride: int,
    nodata_mask: np.ndarray | None = None,
    reference_min_z: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Build a decimated height-field mesh in a local Three.js-friendly frame."""
    height_px, width_px = dem.shape
    rows = np.arange(0, height_px, stride, dtype=np.int32)
    cols = np.arange(0, width_px, stride, dtype=np.int32)
    grid_h, grid_w = len(rows), len(cols)

    dem_sample = dem[np.ix_(rows, cols)].astype(np.float64)
    valid = np.isfinite(dem_sample) & (dem_sample != NODATA)
    if nodata_mask is not None:
        nodata_sample = nodata_mask[np.ix_(rows, cols)]
        valid &= ~nodata_sample

    sample_min = float(np.nanmin(np.where(valid, dem_sample, np.nan)))
    max_z = float(np.nanmax(np.where(valid, dem_sample, np.nan)))
    min_z = sample_min if reference_min_z is None else float(reference_min_z)
    fill_z = sample_min if reference_min_z is None else float(reference_min_z)

    center_x = transform.c + (width_px * transform.a) / 2.0
    center_y = transform.f + (height_px * transform.e) / 2.0

    vertex_count = grid_h * grid_w
    positions = np.zeros(vertex_count * 3, dtype=np.float32)
    uvs = np.zeros(vertex_count * 2, dtype=np.float32)

    idx = 0
    for gi, row in enumerate(rows):
        northing = transform.f + (row + 0.5) * transform.e
        for gj, col in enumerate(cols):
            easting = transform.c + (col + 0.5) * transform.a
            elevation = dem_sample[gi, gj]
            if not valid[gi, gj]:
                elevation = fill_z

            positions[idx] = easting - center_x
            positions[idx + 1] = elevation - min_z
            positions[idx + 2] = -(northing - center_y)

            uvs[idx // 3 * 2] = col / max(width_px - 1, 1)
            uvs[idx // 3 * 2 + 1] = 1.0 - row / max(height_px - 1, 1)
            idx += 3

    indices: list[int] = []
    for gi in range(grid_h - 1):
        for gj in range(grid_w - 1):
            i00 = gi * grid_w + gj
            i10 = gi * grid_w + (gj + 1)
            i01 = (gi + 1) * grid_w + gj
            i11 = (gi + 1) * grid_w + (gj + 1)
            # Winding chosen so surface normals point upward (+Y).
            indices.extend([i00, i10, i01, i10, i11, i01])

    return (
        positions,
        uvs,
        np.asarray(indices, dtype=np.uint32),
        min_z,
        max_z,
    )


def _write_texture(path: Path, rgb: np.ndarray) -> None:
    """Write an H×W×3 uint8 RGB array as JPEG (smaller than PNG for the browser)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 98])


def _texture_stride(width: int, height: int, max_dim: int = MAX_TEXTURE_DIM) -> int:
    longest = max(width, height)
    if longest <= max_dim:
        return 1
    return int(np.ceil(longest / max_dim))


def _parse_gpx_points(path: Path) -> tuple[str, list[tuple[float, float, float | None]]]:
    """Return track name and WGS84 points as (lon, lat, elevation_m | None)."""
    root = ET.parse(path).getroot()
    name = path.stem
    for tag in ("name", f"{{{GPX_NS}}}name"):
        meta_name = root.find(f".//{tag}")
        if meta_name is not None and meta_name.text:
            name = meta_name.text.strip()
            break

    points: list[tuple[float, float, float | None]] = []
    for trkpt in root.iter():
        if trkpt.tag not in ("trkpt", f"{{{GPX_NS}}}trkpt"):
            continue
        lon = float(trkpt.get("lon", "nan"))
        lat = float(trkpt.get("lat", "nan"))
        ele_el = None
        for child in trkpt:
            if child.tag in ("ele", f"{{{GPX_NS}}}ele") and child.text:
                ele_el = float(child.text)
                break
        points.append((lon, lat, ele_el))
    return name, points


def _sample_dem_elevation(
    dem: np.ndarray,
    transform: rasterio.Affine,
    easting: float,
    northing: float,
) -> float | None:
    """Bilinear DEM sample at an LV95 coordinate."""
    col_f = (easting - transform.c) / transform.a - 0.5
    row_f = (northing - transform.f) / transform.e - 0.5
    height_px, width_px = dem.shape
    if col_f < 0 or row_f < 0 or col_f >= width_px - 1 or row_f >= height_px - 1:
        return None

    col0 = int(np.floor(col_f))
    row0 = int(np.floor(row_f))
    col1 = col0 + 1
    row1 = row0 + 1
    tx = col_f - col0
    ty = row_f - row0

    samples = []
    for row, weight_y in ((row0, 1.0 - ty), (row1, ty)):
        for col, weight_x in ((col0, 1.0 - tx), (col1, tx)):
            value = float(dem[row, col])
            if not np.isfinite(value) or value == NODATA:
                return None
            samples.append((value, weight_x * weight_y))
    return float(sum(value * weight for value, weight in samples))


def _discover_gpx_files(root: Path) -> list[Path]:
    sample_dir = root / "data" / "sample"
    if not sample_dir.is_dir():
        return []
    return sorted(sample_dir.glob("*.gpx"))


def _export_gpx_tracks(
    gpx_paths: list[Path],
    *,
    dem: np.ndarray,
    transform: rasterio.Affine,
    center_x: float,
    center_y: float,
    min_z: float,
    bounds: list[float] | None,
    lift_m: float = TRACK_LIFT_M,
) -> list[dict[str, Any]]:
    """Convert GPX tracks to viewer-local coordinates draped on the DEM."""
    if not gpx_paths:
        return []

    to_lv95 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    tracks: list[dict[str, Any]] = []

    for gpx_path in gpx_paths:
        name, wgs_points = _parse_gpx_points(gpx_path)
        if not wgs_points:
            continue

        local_positions: list[float] = []
        for lon, lat, gpx_ele in wgs_points:
            easting, northing = to_lv95.transform(lon, lat)
            if bounds is not None:
                if not (bounds[0] <= easting <= bounds[2] and bounds[1] <= northing <= bounds[3]):
                    continue

            elevation = _sample_dem_elevation(dem, transform, easting, northing)
            if elevation is None and gpx_ele is not None:
                elevation = gpx_ele
            if elevation is None:
                continue

            local_positions.extend(
                [
                    easting - center_x,
                    elevation - min_z + lift_m,
                    -(northing - center_y),
                ],
            )

        if len(local_positions) < 6:
            continue

        try:
            source = str(gpx_path.relative_to(get_project_root()))
        except ValueError:
            source = str(gpx_path)

        tracks.append(
            {
                "name": name,
                "source": source,
                "point_count": len(local_positions) // 3,
                "positions": local_positions,
                "lift_m": lift_m,
            },
        )

    return tracks


def _read_rgb(path: Path, stride: int) -> np.ndarray:
    with rasterio.open(path) as src:
        if stride <= 1:
            data = src.read(indexes=(1, 2, 3))
        else:
            out_h = len(range(0, src.height, stride))
            out_w = len(range(0, src.width, stride))
            data = src.read(
                indexes=(1, 2, 3),
                out_shape=(3, out_h, out_w),
                resampling=rasterio.enums.Resampling.bilinear,
            )
    return np.transpose(data, (1, 2, 0))


def export_tile_viewer_data(
    tile_id: str,
    *,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    stride: int | None = None,
    max_grid_dim: int = MAX_GRID_DIM,
    max_texture_dim: int = MAX_TEXTURE_DIM,
    include_summer: bool = True,
    gpx_paths: list[str | Path] | None = None,
    auto_gpx: bool = True,
) -> dict[str, Any]:
    """Export DEM mesh and ortho textures for the web viewer."""
    config = load_config(config_path)
    paths = tile_paths(config, tile_id)

    if not paths.dem.exists():
        raise FileNotFoundError(f"DEM not found: {paths.dem}")
    if not paths.winter_rgb.exists():
        raise FileNotFoundError(f"Winter ortho not found: {paths.winter_rgb}")

    root = get_project_root()
    out = Path(output_dir) if output_dir else root / "viewer" / "data" / tile_id
    out.mkdir(parents=True, exist_ok=True)

    with rasterio.open(paths.dem) as dem_src:
        dem = dem_src.read(1)
        transform = dem_src.transform
        width_px, height_px = dem_src.width, dem_src.height

    if stride is None:
        stride = _auto_stride(width_px, height_px, max_grid_dim)

    nodata_mask = None
    if paths.nodata_mask.exists():
        with rasterio.open(paths.nodata_mask) as mask_src:
            nodata_mask = mask_src.read(1).astype(bool)

    positions, uvs, indices, min_z, max_z = _build_mesh(
        dem,
        transform=transform,
        stride=stride,
        nodata_mask=nodata_mask,
    )

    center_x = transform.c + (width_px * transform.a) / 2.0
    center_y = transform.f + (height_px * transform.e) / 2.0

    vertex_count = len(positions) // 3
    if vertex_count <= 65535:
        index_bytes = indices.astype(np.uint16).tobytes()
        index_dtype = "uint16"
    else:
        index_bytes = indices.tobytes()
        index_dtype = "uint32"

    positions_path = out / "positions.bin"
    uvs_path = out / "uvs.bin"
    indices_path = out / "indices.bin"
    positions_path.write_bytes(positions.tobytes())
    uvs_path.write_bytes(uvs.tobytes())
    indices_path.write_bytes(index_bytes)

    elevation_models: dict[str, str] = {"base": "positions.bin"}
    snow_surface_meta: dict[str, float] | None = None
    if paths.snow_surface_dem.exists():
        with rasterio.open(paths.snow_surface_dem) as snow_src:
            snow_dem = snow_src.read(1)
        snow_positions, _, _, _, snow_max_z = _build_mesh(
            snow_dem,
            transform=transform,
            stride=stride,
            nodata_mask=nodata_mask,
            reference_min_z=min_z,
        )
        snow_positions_path = out / "positions_snow.bin"
        snow_positions_path.write_bytes(snow_positions.tobytes())
        elevation_models["snow_surface"] = "positions_snow.bin"
        snow_surface_meta = {
            "max_elevation_m": snow_max_z,
            "elevation_range_m": snow_max_z - min_z,
        }

    tex_stride = _texture_stride(width_px, height_px, max_dim=max_texture_dim)
    winter_rgb = _read_rgb(paths.winter_rgb, tex_stride)
    _write_texture(out / "winter.jpg", winter_rgb)

    summer_path: str | None = None
    if include_summer and paths.rgb_summer.exists():
        summer_rgb = _read_rgb(paths.rgb_summer, tex_stride)
        _write_texture(out / "summer.jpg", summer_rgb)
        summer_path = "summer.jpg"

    meta = {
        "tile_id": tile_id,
        "vertex_count": vertex_count,
        "index_count": len(indices),
        "index_dtype": index_dtype,
        "stride": stride,
        "max_texture_dim": max_texture_dim,
        "texture_stride": tex_stride,
        "texture_width": winter_rgb.shape[1],
        "texture_height": winter_rgb.shape[0],
        "source_width": width_px,
        "source_height": height_px,
        "min_elevation_m": min_z,
        "max_elevation_m": max_z,
        "elevation_range_m": max_z - min_z,
        "textures": {
            "winter": "winter.jpg",
            "summer": summer_path,
        },
        "files": {
            "positions": "positions.bin",
            "uvs": "uvs.bin",
            "indices": "indices.bin",
        },
        "elevation_models": elevation_models,
        "has_snow_surface": "snow_surface" in elevation_models,
    }
    if snow_surface_meta is not None:
        meta["snow_surface"] = snow_surface_meta
    bounds: list[float] | None = None
    if paths.metadata.exists():
        with paths.metadata.open(encoding="utf-8") as handle:
            tile_meta = json.load(handle)
        meta["bounds"] = tile_meta.get("bounds")
        meta["crs"] = tile_meta.get("crs")
        raw_bounds = tile_meta.get("bounds")
        if isinstance(raw_bounds, list) and len(raw_bounds) == 4:
            bounds = [float(v) for v in raw_bounds]

    resolved_gpx = [Path(p) for p in gpx_paths] if gpx_paths else []
    if not resolved_gpx and auto_gpx:
        resolved_gpx = _discover_gpx_files(root)
    tracks = _export_gpx_tracks(
        resolved_gpx,
        dem=dem,
        transform=transform,
        center_x=center_x,
        center_y=center_y,
        min_z=min_z,
        bounds=bounds,
    )
    if tracks:
        tracks_path = out / "tracks.json"
        with tracks_path.open("w", encoding="utf-8") as handle:
            json.dump({"tracks": tracks}, handle, indent=2)
        meta["tracks_file"] = "tracks.json"
        meta["track_count"] = len(tracks)

    scene_path = out / "scene.json"
    with scene_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    _update_manifest(out.parent, tile_id)

    return {
        "tile_id": tile_id,
        "output_dir": str(out),
        "scene": str(scene_path),
        "vertex_count": meta["vertex_count"],
        "triangle_count": meta["index_count"] // 3,
        "stride": stride,
        "texture_width": meta["texture_width"],
        "texture_height": meta["texture_height"],
        "texture_stride": tex_stride,
        "max_texture_dim": max_texture_dim,
    }
