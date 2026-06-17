"""Library interface for winter orthophoto generation pipeline.

This module provides programmatic access to the pipeline steps, suitable for
integration into web APIs (e.g., FastAPI + Celery) or other orchestration layers.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

from winter_ortho.pipeline import (
    run_harmonize,
    run_masks,
    run_terrain,
    run_snow_surface,
    run_snow,
    run_render,
    run_qa_step,
    PIPELINE_STEPS,
    PIPELINE_STEPS_SNOW,
)
from winter_ortho.data_prep.region import prepare_region as _prepare_region
from winter_ortho.utils.config import DEFAULT_PROFILE, load_config, load_profile
from winter_ortho.utils.progress import PipelineProgress
from winter_ortho.utils.paths import tile_paths
from winter_ortho.viewer.export import export_tile_viewer_data


class PipelineTask:
    """Represents a running or queued pipeline task.

    This class is used by orchestration layers (Celery, etc.) to track
    progress and retrieve results asynchronously.
    """

    def __init__(self, task_id: str, step_names: List[str]):
        self.task_id = task_id
        self.step_names = step_names
        self.completed_steps: List[str] = []
        self.results: Dict[str, Any] = {}
        self.error: Optional[str] = None

    def advance(self, step_name: str) -> None:
        """Mark a step as completed."""
        self.completed_steps.append(step_name)

    def complete(self, result: Dict[str, Any]) -> None:
        """Mark task as completed with final result."""
        self.results = result

    def fail(self, error: str) -> None:
        """Mark task as failed with error message."""
        self.error = error

    @property
    def is_complete(self) -> bool:
        return self.error is not None or len(self.completed_steps) == len(self.step_names)

    @property
    def progress(self) -> float:
        if self.is_complete:
            return 1.0
        return len(self.completed_steps) / len(self.step_names)


class LibraryProgress(PipelineProgress):
    """PipelineProgress implementation that reports to a callable.

    This allows the library to integrate with any progress reporting system
    (Redis Pub/Sub, websockets, etc.).
    """

    def __init__(self, on_step: Optional[Callable[[str, int, int], None]] = None):
        self.on_step = on_step
        super().__init__()

    def advance(self, title: str) -> None:
        """Called at the start of each pipeline step."""
        if self.on_step:
            self.on_step(title, 0, 0)

    def step_begin(self, current: int, total: int, title: str, detail: str = "") -> None:
        """Called when a step begins."""
        if self.on_step:
            self.on_step(title, current, total)

    def step_end(self, title: str, summary: str) -> None:
        """Called when a step completes."""
        if self.on_step:
            self.on_step(title, 0, 0)


@dataclass
class PrepareRegionResult:
    """Result of prepare_region library call."""
    config_path: str
    region_dir: str
    tile_id: str
    run_command: str


@dataclass
class ViewerExportResult:
    """Result of viewer_export library call."""
    output_dir: str
    vertex_count: int
    triangle_count: int
    stride: int
    texture_width: int
    texture_height: int
    texture_stride: int
    max_texture_dim: int


def run_pipeline_task(
    *,
    tile_id: str,
    profile_name: str = DEFAULT_PROFILE,
    config_path: Optional[str] = None,
    steps: List[str] = None,
    on_step: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, Any]:
    """Run a sequence of pipeline steps programmatically.

    Args:
        tile_id: Tile identifier from config
        profile_name: Rendering profile name
        config_path: Path to config file (default: uses wintermaker default)
        steps: List of step names to run. If None, runs all steps.
               Valid steps: harmonize, masks, terrain, snow_surface, snow, render, qa
        on_step: Optional callback(step_title, current, total) for progress

    Returns:
        Dictionary with all step results keyed by step name
    """
    if steps is None:
        steps = [s[0] for s in PIPELINE_STEPS]

    step_map = {
        "harmonize": lambda: run_harmonize(tile_id, config_path),
        "masks": lambda: run_masks(tile_id, config_path),
        "terrain": lambda: run_terrain(tile_id, config_path),
        "snow_surface": lambda: run_snow_surface(tile_id, profile_name, config_path),
        "snow": lambda: run_snow(tile_id, profile_name, config_path),
        "render": lambda: run_render(tile_id, profile_name, config_path),
        "qa": lambda: run_qa_step(tile_id, config_path),
    }

    # Filter to requested steps
    available_steps = {s[0]: (s[1], step_map[s[0]]) for s in PIPELINE_STEPS}
    requested = [(k, available_steps[k][0], available_steps[k][1]) for k in steps if k in available_steps]

    config = load_config(config_path)
    paths = tile_paths(config, tile_id)

    progress = LibraryProgress(on_step=on_step)
    results: Dict[str, Any] = {"tile_id": tile_id, "profile": profile_name}

    for step_key, step_title, step_fn in requested:
        try:
            result = step_fn()
            results[step_key] = result
        except Exception as e:
            results[step_key] = {"error": str(e)}
            raise

    return results


def run_all_async(
    *,
    tile_id: str,
    profile_name: str = DEFAULT_PROFILE,
    config_path: Optional[str] = None,
    on_step: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, Any]:
    """Run full pipeline asynchronously (returns immediately).

    Use this for Celery tasks. The actual execution happens in the
    background; this function returns a result dict that can be
    serialized.

    Args:
        tile_id: Tile identifier
        profile_name: Rendering profile name
        config_path: Path to config file
        on_step: Optional callback for progress updates

    Returns:
        Final result dictionary (same as run_all but without progress)
    """
    return run_pipeline_task(
        tile_id=tile_id,
        profile_name=profile_name,
        config_path=config_path,
        on_step=on_step,
    )


def prepare_region_async(
    *,
    name: str,
    extent: str,
    base_config: Optional[str] = None,
    tlm_source: Optional[str] = None,
    dem_year: int = 2023,
    wmts_zoom: Optional[int] = None,
    skip_ortho: bool = False,
    skip_dem: bool = False,
    skip_tlm: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
) -> PrepareRegionResult:
    """Prepare a region (download data + create config) asynchronously.

    Args:
        name: Region name
        extent: Bounding box minx,miny,maxx,maxy in EPSG:2056
        base_config: Base config to copy (default: wintermaker default)
        tlm_source: Master swissTLM3D GeoPackage
        dem_year: Preferred swissALTI3D vintage
        wmts_zoom: Override WMTS zoom
        skip_ortho: Skip orthophoto download
        skip_dem: Skip DEM download
        skip_tlm: Skip TLM3D extraction
        on_progress: Optional callback(message) for progress

    Returns:
        PrepareRegionResult with config_path, region_dir, tile_id, run_command
    """
    # Convert callback to report function
    def report(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    result = _prepare_region(
        name=name,
        extent=extent,
        base_config=base_config,
        tlm_source=tlm_source,
        dem_year=dem_year,
        wmts_zoom=wmts_zoom,
        skip_ortho=skip_ortho,
        skip_dem=skip_dem,
        skip_tlm=skip_tlm,
        progress=report,
    )
    run_cmd = result.get("run_command", "")
    if not isinstance(run_cmd, str):
        run_cmd = str(run_cmd)

    return PrepareRegionResult(
        config_path=str(result["config"]),
        region_dir=str(result["region_dir"]),
        tile_id=str(result["tile_id"]),
        run_command=run_cmd,
    )


def export_viewer_async(
    *,
    tile_id: str,
    config_path: Optional[str] = None,
    stride: int = 2,
    max_texture_dim: int = 16384,
) -> ViewerExportResult:
    """Export 3D viewer data asynchronously.

    Args:
        tile_id: Tile identifier
        config_path: Path to config file
        stride: Mesh decimation stride
        max_texture_dim: Max orthophoto texture edge length in pixels

    Returns:
        ViewerExportResult with mesh and texture info
    """
    result = export_tile_viewer_data(
        tile_id,
        config_path=str(config_path) if config_path else None,
        stride=stride,
        max_texture_dim=max_texture_dim,
    )

    return ViewerExportResult(
        output_dir=str(result["output_dir"]),
        vertex_count=result["vertex_count"],
        triangle_count=result["triangle_count"],
        stride=result["stride"],
        texture_width=result["texture_width"],
        texture_height=result["texture_height"],
        texture_stride=result["texture_stride"],
        max_texture_dim=result["max_texture_dim"],
    )


# Convenience aliases for common operations
run_snow_pipeline = run_all_async
