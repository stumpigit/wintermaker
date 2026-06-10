from __future__ import annotations

import pytest

from winter_ortho import pipeline


@pytest.fixture(autouse=True)
def _chdir(monkeypatch: pytest.MonkeyPatch, synthetic_tile: dict) -> None:
    monkeypatch.chdir(synthetic_tile["root"])


def test_qa_runs_after_full_pipeline(synthetic_tile: dict) -> None:
    tile_id = synthetic_tile["tile_id"]
    config_path = str(synthetic_tile["config_path"])

    result = pipeline.run_all(tile_id, "davos", config_path)

    assert "qa" in result
    assert "checks" in result["qa"]
    assert "building_edges" in result["qa"]["checks"]
    assert "water_boundary" in result["qa"]["checks"]
