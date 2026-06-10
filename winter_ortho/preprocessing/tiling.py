from __future__ import annotations

from typing import Any

from winter_ortho.utils.raster import TargetGrid


def get_tile_bbox(config: dict[str, Any], tile_id: str) -> tuple[float, float, float, float]:
    tiles = config.get("tiles", {})
    if tile_id not in tiles:
        raise KeyError(f"Tile '{tile_id}' not defined in config tiles section")
    bbox = tiles[tile_id]["bbox"]
    if len(bbox) != 4:
        raise ValueError(f"Tile bbox for '{tile_id}' must have 4 values")
    return tuple(float(v) for v in bbox)


def get_tile_grid(config: dict[str, Any], tile_id: str) -> TargetGrid:
    bbox = get_tile_bbox(config, tile_id)
    return TargetGrid.from_bbox(
        bbox=bbox,
        resolution_m=float(config["resolution_m"]),
        crs=config["crs"],
    )
