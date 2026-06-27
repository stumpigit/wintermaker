"""Library interface for winter orthophoto generation pipeline.

This module provides programmatic access to the pipeline steps, suitable for
integration into web APIs (e.g., FastAPI + Celery) or other orchestration layers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)

from winter_ortho.data_prep.region import prepare_region as _prepare_region
from winter_ortho.utils.config import DEFAULT_PROFILE
from winter_ortho.utils.progress import PipelineProgress
from winter_ortho.viewer.export import export_tile_viewer_data


def _pipeline_imports():
    """Lazy import pipeline dependencies.

    The full pipeline (masks / vector clipping) pulls in optional heavy deps like
    geopandas. Viewer export should still work without them, so we import the
    pipeline only when a caller actually runs it.
    """

    from winter_ortho.pipeline import (  # type: ignore
        _run_pipeline_steps,
        run_all,
        run_all_snow,
        PIPELINE_STEPS,
        PIPELINE_STEPS_SNOW,
    )

    return _run_pipeline_steps, run_all, run_all_snow, PIPELINE_STEPS, PIPELINE_STEPS_SNOW


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
        self._current = 0
        self._total = 0
        self._title = ""
        super().__init__(verbose=False)

    def advance(self, title: str) -> None:
        """Called at the start of each pipeline step."""
        if self.on_step:
            self.on_step(title, 0, 0)

    def step_begin(self, current: int, total: int, title: str, detail: str = "") -> None:
        """Called when a step begins."""
        self._current = current
        self._total = total
        self._title = title
        msg = f"[{current}/{total}] {title}"
        if detail:
            msg += f" — {detail}"
        logger.info(msg)
        if self.on_step:
            self.on_step(title, current, total)

    def step_end(self, title: str, summary: str = "") -> None:
        """Called when a step completes."""
        msg = f"✓ {title}"
        if summary:
            msg += f" — {summary}"
        logger.info(msg)

    def substep(self, message: str) -> None:
        logger.info("  → %s", message)
        if self.on_step and self._total > 0:
            self.on_step(f"→ {message}", self._current, self._total)

    def info(self, message: str) -> None:
        logger.info("  i %s", message)
        if self.on_step and self._total > 0:
            self.on_step(f"i {message}", self._current, self._total)

    def warn(self, message: str) -> None:
        logger.warning("  ! %s", message)
        if self.on_step and self._total > 0:
            self.on_step(f"! {message}", self._current, self._total)


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
    # Import full pipeline only when used (may require geopandas, etc.)
    (
        _run_pipeline_steps,
        run_all,
        run_all_snow,
        PIPELINE_STEPS,
        PIPELINE_STEPS_SNOW,
    ) = _pipeline_imports()
    from winter_ortho.pipeline import (  # type: ignore
        run_harmonize,
        run_masks,
        run_terrain,
        run_snow_surface,
        run_snow,
        run_render,
        run_qa_step,
    )

    if steps is None:
        steps = [s[0] for s in PIPELINE_STEPS]

    if steps == [s[0] for s in PIPELINE_STEPS]:
        return run_all(
            tile_id,
            profile_name,
            config_path,
            progress=LibraryProgress(on_step=on_step),
        )
    if steps == [s[0] for s in PIPELINE_STEPS_SNOW]:
        return run_all_snow(
            tile_id,
            profile_name,
            config_path,
            progress=LibraryProgress(on_step=on_step),
        )

    progress = LibraryProgress(on_step=on_step)
    step_defs = {key: (title, detail) for key, title, detail in PIPELINE_STEPS}
    step_fns = {
        "harmonize": lambda: run_harmonize(tile_id, config_path, progress=progress),
        "masks": lambda: run_masks(tile_id, config_path, progress=progress),
        "terrain": lambda: run_terrain(tile_id, config_path, progress=progress),
        "snow_surface": lambda: run_snow_surface(
            tile_id, profile_name, config_path, progress=progress
        ),
        "snow": lambda: run_snow(tile_id, profile_name, config_path, progress=progress),
        "render": lambda: run_render(tile_id, profile_name, config_path, progress=progress),
        "qa": lambda: run_qa_step(tile_id, config_path, progress=progress),
    }

    selected_steps = [
        (key, step_defs[key][0], step_defs[key][1]) for key in steps if key in step_defs
    ]
    selected_fns = [step_fns[key] for key in steps if key in step_fns]

    return _run_pipeline_steps(
        tile_id=tile_id,
        profile_name=profile_name,
        config_path=config_path,
        steps=selected_steps,
        step_fns=selected_fns,
        progress=progress,
    )


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
    _, run_all, _, _, _ = _pipeline_imports()
    return run_all(
        tile_id,
        profile_name,
        config_path,
        progress=LibraryProgress(on_step=on_step),
    )


def run_snow_async(
    *,
    tile_id: str,
    profile_name: str = DEFAULT_PROFILE,
    config_path: Optional[str] = None,
    on_step: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, Any]:
    """Run snow-only pipeline (snow_surface → snow → render → qa)."""
    _, _, run_all_snow, _, _ = _pipeline_imports()
    return run_all_snow(
        tile_id,
        profile_name,
        config_path,
        progress=LibraryProgress(on_step=on_step),
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
    gpx_paths: Optional[list[str]] = None,
    auto_gpx: bool = True,
) -> ViewerExportResult:
    """Export 3D viewer data asynchronously.

    Args:
        tile_id: Tile identifier
        config_path: Path to config file
        stride: Mesh decimation stride
        max_texture_dim: Max orthophoto texture edge length in pixels
        gpx_paths: Optional list of GPX file paths to include in the viewer
        auto_gpx: Discover sample GPX files when gpx_paths is empty

    Returns:
        ViewerExportResult with mesh and texture info
    """
    result = export_tile_viewer_data(
        tile_id,
        config_path=str(config_path) if config_path else None,
        stride=stride,
        max_texture_dim=max_texture_dim,
        gpx_paths=gpx_paths,
        auto_gpx=auto_gpx,
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
run_snow_pipeline = run_snow_async
