"""Winter orthophoto generation pipeline.

This package provides both CLI and programmatic access to the winter
orthophoto generation pipeline.

Library API:
    from winter_ortho import library
    library.run_all_async(...)
    library.prepare_region_async(...)
"""

__version__ = "0.1.0"

from winter_ortho.library import (
    run_all_async,
    run_snow_async,
    run_snow_pipeline,
    prepare_region_async,
    export_viewer_async,
    run_pipeline_task,
    PrepareRegionResult,
    ViewerExportResult,
)

__all__ = [
    "__version__",
    "run_all_async",
    "run_snow_async",
    "run_snow_pipeline",
    "prepare_region_async",
    "export_viewer_async",
    "run_pipeline_task",
    "PrepareRegionResult",
    "ViewerExportResult",
]
