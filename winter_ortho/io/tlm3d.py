from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import box


def load_tlm_layer(
    source_path: str | Path,
    *,
    target_crs: str,
    bbox: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame:
    path = Path(source_path)
    if not path.exists():
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(target_crs)
    else:
        gdf = gdf.to_crs(target_crs)

    if bbox is not None:
        clip_box = box(*bbox)
        gdf = gdf[gdf.intersects(clip_box)].copy()
        gdf["geometry"] = gdf.geometry.intersection(clip_box)

    return gdf.reset_index(drop=True)
