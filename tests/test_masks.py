from __future__ import annotations

import numpy as np
import pytest

from winter_ortho import pipeline
from winter_ortho.utils.config import load_class_rules, load_config
from winter_ortho.utils.paths import tile_paths


@pytest.fixture(autouse=True)
def _chdir(monkeypatch: pytest.MonkeyPatch, synthetic_tile: dict) -> None:
    monkeypatch.chdir(synthetic_tile["root"])


def test_build_masks_creates_all_layers(synthetic_tile: dict) -> None:
    tile_id = synthetic_tile["tile_id"]
    config_path = str(synthetic_tile["config_path"])

    pipeline.run_harmonize(tile_id, config_path)
    result = pipeline.run_masks(tile_id, config_path)

    assert result["mask_count"] == 9
    paths = tile_paths(load_config(config_path), tile_id)
    assert paths.tlm_masks.exists()
    assert (paths.intermediate_dir / "water_mask.tif").exists()
    assert (paths.intermediate_dir / "building_mask.tif").exists()


def test_mask_priority_resolves_overlap(synthetic_tile: dict) -> None:
    tile_id = synthetic_tile["tile_id"]
    config_path = str(synthetic_tile["config_path"])
    pipeline.run_harmonize(tile_id, config_path)
    pipeline.run_masks(tile_id, config_path)

    paths = tile_paths(load_config(config_path), tile_id)
    class_rules = load_class_rules()
    building, _ = __import__("winter_ortho.utils.raster", fromlist=["read_raster"]).read_raster(
        str(paths.intermediate_dir / "building_mask.tif")
    )
    labeled, _ = __import__("winter_ortho.utils.raster", fromlist=["read_raster"]).read_raster(
        str(paths.tlm_masks)
    )
    building_val = class_rules["mask_values"]["building_mask"]
    assert (labeled[building > 0] == building_val).all()
