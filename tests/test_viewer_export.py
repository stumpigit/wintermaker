from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rasterio import Affine

from winter_ortho.viewer.export import _auto_stride, _build_mesh


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
