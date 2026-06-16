from __future__ import annotations

import copy
import re
import sys
from pathlib import Path
from typing import Callable

import yaml

from winter_ortho.data_prep.dem import fetch_swissalti3d_mosaic
from winter_ortho.data_prep.wmts import (
    describe_tile_range,
    fetch_swissimage_mosaic,
    parse_extent,
    tile_range_for_bbox,
    zoom_for_resolution,
)
from winter_ortho.utils.config import DEFAULT_PROFILE
from winter_ortho.utils.paths import get_project_root

DEFAULT_TLM_SOURCE = "data/raw/swisstlm/SWISSTLM3D_2026_LV95_LN02.gpkg"


def _validate_name(name: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
        raise ValueError(
            "name must start with a letter and contain only lowercase letters, digits, _ or -"
        )
    return name


def tile_id_for_name(name: str) -> str:
    return f"{name}_001"


def region_raw_dir(root: Path, name: str) -> Path:
    return root / "data" / "raw" / "regions" / name


def region_config_path(root: Path, name: str) -> Path:
    return root / "config" / "regions" / f"{name}.yaml"


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_region_config(
    *,
    root: Path,
    name: str,
    bbox: tuple[float, float, float, float],
    base_config_path: Path,
) -> Path:
    region_config = copy.deepcopy(_load_yaml(base_config_path))
    region_dir = region_raw_dir(root, name)
    rel = lambda *parts: "/".join(("data", "raw", "regions", name, *parts))

    region_config["paths"]["orthophoto"] = rel("orthophoto", "summer_rgb.tif")
    region_config["paths"]["dem"] = rel("dem", "dem.tif")
    region_config["paths"]["tlm3d"] = {
        layer: rel("tlm3d", f"{layer}.gpkg")
        for layer in (
            "buildings",
            "roads",
            "paths",
            "water",
            "forest",
            "settlement",
            "landcover",
        )
    }

    tile_id = tile_id_for_name(name)
    region_config["tiles"] = {tile_id: {"bbox": list(bbox)}}

    config_path = region_config_path(root, name)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(region_config, handle, sort_keys=False, allow_unicode=True)
    return config_path


def prepare_region(
    *,
    name: str,
    extent: str,
    base_config: str | Path | None = None,
    tlm_source: str | Path | None = None,
    dem_year: int = 2023,
    wmts_zoom: int | None = None,
    skip_ortho: bool = False,
    skip_dem: bool = False,
    skip_tlm: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    name = _validate_name(name)
    bbox = parse_extent(extent)
    root = get_project_root()
    base_config_path = Path(base_config) if base_config else root / "config" / "default.yaml"
    region_dir = region_raw_dir(root, name)
    ortho_path = region_dir / "orthophoto" / "summer_rgb.tif"
    dem_path = region_dir / "dem" / "dem.tif"
    tlm_dir = region_dir / "tlm3d"

    result: dict[str, object] = {
        "name": name,
        "tile_id": tile_id_for_name(name),
        "bbox": bbox,
        "region_dir": str(region_dir),
    }

    base_config = _load_yaml(base_config_path)
    resolution_m = float(base_config.get("resolution_m", 2.0))

    if not skip_ortho:
        ortho_path.parent.mkdir(parents=True, exist_ok=True)
        if progress:
            zoom = wmts_zoom if wmts_zoom is not None else zoom_for_resolution(resolution_m)
            tile_info = describe_tile_range(tile_range_for_bbox(bbox, zoom=zoom))
            progress(
                "WMTS SWISSIMAGE: "
                f"zoom {tile_info['zoom']} "
                f"({tile_info['native_resolution_m']:.2f} m), "
                f"{tile_info['tile_count']} Kacheln "
                f"col {tile_info['col_range'][0]}–{tile_info['col_range'][1]}, "
                f"row {tile_info['row_range'][0]}–{tile_info['row_range'][1]}"
            )
        result["orthophoto"] = fetch_swissimage_mosaic(
            bbox,
            str(ortho_path),
            resolution_m=resolution_m,
            zoom=wmts_zoom,
            progress=progress,
        )

    if not skip_dem:
        result["dem"] = fetch_swissalti3d_mosaic(
            bbox,
            dem_path,
            year=dem_year,
            progress=progress,
        )

    if not skip_tlm:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from scripts.extract_tlm3d import DEFAULT_SOURCE, extract_tlm3d

        source = Path(tlm_source) if tlm_source else root / DEFAULT_SOURCE
        if not source.exists():
            raise FileNotFoundError(
                f"swissTLM3D source not found: {source}. "
                "Place the national GeoPackage or pass --tlm-source."
            )
        if progress:
            progress("TLM3D: clipping vector layers")
        result["tlm3d"] = extract_tlm3d(
            source=source,
            output_dir=tlm_dir,
            bbox=bbox,
        )

    config_path = write_region_config(
        root=root,
        name=name,
        bbox=bbox,
        base_config_path=base_config_path,
    )
    result["config"] = str(config_path)
    result["profile"] = DEFAULT_PROFILE
    result["run_command"] = (
        f"winter-ortho run-all --tile-id {tile_id_for_name(name)} "
        f"--profile {DEFAULT_PROFILE} --config {config_path.relative_to(root)}"
    )
    return result
