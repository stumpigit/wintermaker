from pathlib import Path

from winter_ortho.utils.config import load_config
from winter_ortho.utils.paths import get_project_root


def test_get_project_root_from_region_config_dir():
    root = get_project_root(Path("config/regions"))
    assert (root / "config" / "default.yaml").exists()
    assert (root / "pyproject.toml").exists()


def test_load_config_resolves_region_paths_from_project_root():
    config = load_config("config/regions/demo_test.yaml")
    ortho = Path(config["paths"]["orthophoto"])
    assert ortho == get_project_root() / "data/raw/regions/demo_test/orthophoto/summer_rgb.tif"
    assert ortho.exists()
