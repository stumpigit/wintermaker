from __future__ import annotations

from winter_ortho.utils.raster import TargetGrid

# Rough float32 bytes per full raster layer
_BYTES_PER_PIXEL = 4


def grid_pixel_count(grid: TargetGrid) -> int:
    return grid.width * grid.height


def estimate_layer_mb(grid: TargetGrid, bands: int = 1) -> float:
    return grid_pixel_count(grid) * _BYTES_PER_PIXEL * bands / (1024**2)


def format_grid_summary(grid: TargetGrid) -> str:
    px = grid_pixel_count(grid)
    span_x = grid.bounds[2] - grid.bounds[0]
    span_y = grid.bounds[3] - grid.bounds[1]
    return (
        f"{grid.width}×{grid.height} px ({px / 1e6:.1f} Mpx), "
        f"{span_x / 1000:.1f}×{span_y / 1000:.1f} km @ {grid.transform.a} m"
    )


def check_grid_memory(
    grid: TargetGrid,
    *,
    bands: int = 10,
    warn_mb: float = 1500,
    hard_limit_mb: float = 8000,
) -> list[str]:
    """Return warning messages if the grid is likely to exhaust RAM."""
    messages: list[str] = []
    layer_mb = estimate_layer_mb(grid, 1)
    stack_mb = estimate_layer_mb(grid, bands)
    px = grid_pixel_count(grid)

    if layer_mb > warn_mb:
        messages.append(
            f"Large raster: {format_grid_summary(grid)} — "
            f"~{layer_mb:.0f} MB per layer, ~{stack_mb:.0f} MB for {bands} bands. "
            "Consider a smaller bbox, coarser resolution_m (e.g. 1.0), "
            "or tile_size_m-aligned clips (~512 m)."
        )
    if stack_mb > hard_limit_mb:
        messages.append(
            f"Raster may exceed available RAM (~{stack_mb:.0f} MB for terrain stack). "
            "Pipeline was likely OOM-killed (exit 137) at this size."
        )
    if px > 100_000_000:
        messages.append(
            f"{px / 1e6:.0f} Mpx total — terrain filters run on the full extent in memory."
        )
    return messages
