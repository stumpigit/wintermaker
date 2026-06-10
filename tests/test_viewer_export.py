from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rasterio import Affine

from winter_ortho.viewer.export import (
    _auto_stride,
    _build_mesh,
    _export_gpx_tracks,
    _parse_gpx_points,
    _texture_stride,
)


def test_texture_stride_respects_max_dim() -> None:
    assert _texture_stride(4000, 4000) == 4
    assert _texture_stride(4000, 4000, max_dim=2048) == 2
    assert _texture_stride(4000, 4000, max_dim=4096) == 1


def test_auto_stride_scales_large_rasters() -> None:
    assert _auto_stride(2000, 2000) == 8
    assert _auto_stride(4000, 4000) == 16
    assert _auto_stride(256, 200) == 1


def test_build_mesh_produces_valid_geometry() -> None:
    dem = np.linspace(1000, 1100, 100 * 100, dtype=np.float32).reshape(100, 100)
    transform = Affine(2.0, 0.0, 2788000.0, 0.0, -2.0, 1185000.0)
    positions, uvs, indices, min_z, max_z = _build_mesh(
        dem,
        transform=transform,
        stride=2,
    )
    assert min_z == pytest.approx(1000.0, rel=0.01)
    assert max_z >= 1090.0
    assert positions.shape[0] == (50 * 50) * 3
    assert uvs.shape[0] == (50 * 50) * 2
    assert len(indices) > 0
    assert indices.max() < positions.shape[0] // 3


def test_build_mesh_reference_min_z_aligns_vertex_frames() -> None:
    dem = np.full((20, 20), 1000.0, dtype=np.float32)
    dem[5:10, 5:10] = 1005.0
    snow = dem + 2.0
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 20.0)

    base_pos, _, _, min_z, _ = _build_mesh(dem, transform=transform, stride=2)
    snow_pos, _, _, _, snow_max = _build_mesh(
        snow,
        transform=transform,
        stride=2,
        reference_min_z=min_z,
    )

    assert snow_max == pytest.approx(1007.0)
    for vi in range(base_pos.shape[0] // 3):
        assert base_pos[vi * 3] == pytest.approx(snow_pos[vi * 3])
        assert base_pos[vi * 3 + 2] == pytest.approx(snow_pos[vi * 3 + 2])
        assert snow_pos[vi * 3 + 1] >= base_pos[vi * 3 + 1]


def test_parse_gpx_points_reads_track() -> None:
    gpx = Path("data/sample/Sentischhorn - MAIN Route 22276360.gpx")
    if not gpx.exists():
        pytest.skip("sample GPX not available")

    name, points = _parse_gpx_points(gpx)
    assert "Sentischhorn" in name
    assert len(points) == 120
    assert points[0][0] == pytest.approx(9.919489, abs=1e-4)


def test_export_gpx_tracks_projects_into_tile_bounds() -> None:
    gpx = Path("data/sample/Sentischhorn - MAIN Route 22276360.gpx")
    if not gpx.exists():
        pytest.skip("sample GPX not available")

    dem = np.full((100, 100), 2000.0, dtype=np.float32)
    transform = Affine(4.0, 0.0, 2788000.0, 0.0, -4.0, 1185000.0)
    bounds = [2788000.0, 1181000.0, 2792000.0, 1185000.0]
    tracks = _export_gpx_tracks(
        [gpx],
        dem=dem,
        transform=transform,
        center_x=2790000.0,
        center_y=1183000.0,
        min_z=1800.0,
        bounds=bounds,
    )
    assert len(tracks) == 1
    assert tracks[0]["point_count"] > 10
    positions = tracks[0]["positions"]
    y_values = positions[1::3]
    assert all(y > tracks[0]["lift_m"] for y in y_values)


def test_export_tile_viewer_data(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    from winter_ortho.viewer.export import export_tile_viewer_data

    result = export_tile_viewer_data(
        "demo_test_001",
        config_path="config/regions/demo_test.yaml",
        output_dir=tmp_path / "demo_test_001",
        stride=16,
    )
    scene = json.loads((tmp_path / "demo_test_001" / "scene.json").read_text())
    assert scene["tile_id"] == "demo_test_001"
    assert scene["index_dtype"] == "uint16"
    assert (tmp_path / "demo_test_001" / "winter.jpg").exists()
    assert result["triangle_count"] > 0
    if Path("data/sample/Sentischhorn - MAIN Route 22276360.gpx").exists():
        assert scene.get("tracks_file") == "tracks.json"
        assert (tmp_path / "demo_test_001" / "tracks.json").exists()
