from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from winter_ortho.io.dem import load_and_align_dem
from winter_ortho.io.orthophoto import load_and_align_orthophoto
from winter_ortho.preprocessing.tiling import get_tile_grid
from winter_ortho.utils.grid_stats import check_grid_memory, format_grid_summary
from winter_ortho.utils.paths import TilePaths
from winter_ortho.utils.progress import PipelineProgress
from winter_ortho.utils.raster import alignment_report, write_cog


def harmonize_tile(
    config: dict[str, Any],
    paths: TilePaths,
    *,
    progress: PipelineProgress | None = None,
) -> dict[str, Any]:
    grid = get_tile_grid(config, paths.tile_id)
    harmonize_cfg = config.get("harmonize", {})

    ortho_path = Path(config["paths"]["orthophoto"])
    dem_path = Path(config["paths"]["dem"])
    if not ortho_path.exists():
        raise FileNotFoundError(f"Orthophoto not found: {ortho_path}")
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    if progress:
        progress.substep(f"Grid: {format_grid_summary(grid)}")
        for msg in check_grid_memory(grid):
            progress.warn(msg)
        progress.substep(f"Reprojecting orthophoto: {ortho_path.name}")

    rgb, ortho_valid = load_and_align_orthophoto(
        ortho_path,
        grid,
        resampling=harmonize_cfg.get("orthophoto_resampling", "bilinear"),
    )
    if progress:
        progress.substep(f"Reprojecting DEM: {dem_path.name}")

    dem, dem_valid = load_and_align_dem(
        dem_path,
        grid,
        resampling=harmonize_cfg.get("dem_resampling", "bilinear"),
    )

    nodata_mask = ~(ortho_valid & dem_valid)
    if progress:
        progress.substep(
            f"Writing rgb_summer.tif, dem_2m.tif, nodata_mask.tif "
            f"(nodata {nodata_mask.mean():.1%})"
        )
    write_cog(
        str(paths.rgb_summer),
        rgb,
        transform=grid.transform,
        crs=grid.crs,
        nodata=None,
    )
    write_cog(
        str(paths.dem),
        dem.astype(np.float32),
        transform=grid.transform,
        crs=grid.crs,
        nodata=-9999.0,
    )
    write_cog(
        str(paths.nodata_mask),
        nodata_mask.astype(np.uint8),
        transform=grid.transform,
        crs=grid.crs,
        nodata=None,
    )

    ref_meta = {
        "transform": grid.transform,
        "width": grid.width,
        "height": grid.height,
        "crs": grid.crs,
    }
    dem_meta = dict(ref_meta)
    alignment = alignment_report(ref_meta, dem_meta)

    metadata = {
        "tile_id": paths.tile_id,
        "crs": grid.crs,
        "transform": list(grid.transform),
        "width": grid.width,
        "height": grid.height,
        "bounds": list(grid.bounds),
        "resolution_m": config["resolution_m"],
        "source_paths": {
            "orthophoto": str(ortho_path.resolve()),
            "dem": str(dem_path.resolve()),
        },
        "alignment": alignment,
        "nodata_fraction": float(nodata_mask.mean()),
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    paths.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
