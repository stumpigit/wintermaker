from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio

from winter_ortho.utils.config import load_config
from winter_ortho.utils.paths import get_project_root, tile_paths

NODATA = -9999.0
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

    min_z = float(np.nanmin(np.where(valid, dem_sample, np.nan)))
    max_z = float(np.nanmax(np.where(valid, dem_sample, np.nan)))

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
                elevation = min_z

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
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 94])


def _texture_stride(width: int, height: int, max_dim: int = MAX_TEXTURE_DIM) -> int:
    longest = max(width, height)
    if longest <= max_dim:
        return 1
    return int(np.ceil(longest / max_dim))


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
    include_summer: bool = True,
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

    tex_stride = _texture_stride(width_px, height_px)
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
    }
    if paths.metadata.exists():
        with paths.metadata.open(encoding="utf-8") as handle:
            tile_meta = json.load(handle)
        meta["bounds"] = tile_meta.get("bounds")
        meta["crs"] = tile_meta.get("crs")

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
    }
