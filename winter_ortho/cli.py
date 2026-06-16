from __future__ import annotations

import http.server
import json
import subprocess
import sys
import webbrowser
from functools import partial
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.rule import Rule

from winter_ortho import pipeline
from winter_ortho.data_prep.region import prepare_region
from winter_ortho.pipeline import PIPELINE_STEPS
from winter_ortho.utils.paths import get_project_root
from winter_ortho.utils.progress import PipelineProgress
from winter_ortho.viewer.export import export_tile_viewer_data

app = typer.Typer(help="Winter orthophoto generation pipeline")
console = Console()

VIEWER_EXPORT_STRIDE = 2
VIEWER_EXPORT_MAX_TEXTURE_DIM = 16384


def _progress(quiet: bool) -> PipelineProgress:
    return PipelineProgress(console=console, verbose=not quiet)


def _export_viewer_after_pipeline(
    tile_id: str,
    config: Path | None,
    *,
    quiet: bool,
) -> dict:
    progress = _progress(quiet)
    if not quiet:
        progress.info(
            "Viewer-Export "
            f"(stride={VIEWER_EXPORT_STRIDE}, max-texture-dim={VIEWER_EXPORT_MAX_TEXTURE_DIM})"
        )
    result = export_tile_viewer_data(
        tile_id,
        config_path=str(config) if config else None,
        stride=VIEWER_EXPORT_STRIDE,
        max_texture_dim=VIEWER_EXPORT_MAX_TEXTURE_DIM,
    )
    if quiet:
        console.print(f"viewer-export OK → {result['output_dir']}")
    else:
        console.print(
            f"[green]Viewer-Export:[/green] {result['output_dir']}\n"
            f"  {result['vertex_count']:,} Vertices, "
            f"{result['triangle_count']:,} Dreiecke (stride={result['stride']})\n"
            f"  Textur {result['texture_width']}×{result['texture_height']} px "
            f"(stride={result['texture_stride']}, max={result['max_texture_dim']})"
        )
    return result


def _run_step(
    *,
    step_key: str,
    title: str,
    tile_id: str,
    quiet: bool,
    fn,
    profile: str | None = None,
) -> dict:
    progress = _progress(quiet)
    progress.header(title, tile_id=tile_id, profile=profile)
    progress.step_begin(1, 1, title)
    result = fn(progress)
    if not quiet:
        summary = pipeline._step_summary(step_key, result)  # noqa: SLF001
        progress.step_end(title, summary)
    return result


@app.command()
def harmonize(
    tile_id: str = typer.Option(..., help="Tile identifier from config"),
    config: Optional[Path] = typer.Option(None, help="Path to default.yaml"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
) -> None:
    """Harmonize orthophoto and DEM to a common LV95 grid."""
    result = _run_step(
        step_key="harmonize",
        title="Datenharmonisierung",
        tile_id=tile_id,
        quiet=quiet,
        fn=lambda p: pipeline.run_harmonize(tile_id, str(config) if config else None, progress=p),
    )
    if quiet:
        console.print(f"harmonize OK → {result['width']}×{result['height']} px")


@app.command()
def masks(
    tile_id: str = typer.Option(..., help="Tile identifier"),
    config: Optional[Path] = typer.Option(None, help="Path to default.yaml"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
) -> None:
    """Build geometric masks from swissTLM3D vectors."""
    result = _run_step(
        step_key="masks",
        title="Geometrische Masken",
        tile_id=tile_id,
        quiet=quiet,
        fn=lambda p: pipeline.run_masks(tile_id, str(config) if config else None, progress=p),
    )
    if quiet:
        console.print(f"masks OK → {result['mask_count']} masks")


@app.command()
def terrain(
    tile_id: str = typer.Option(..., help="Tile identifier"),
    config: Optional[Path] = typer.Option(None, help="Path to default.yaml"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
) -> None:
    """Compute terrain features from aligned DEM."""
    result = _run_step(
        step_key="terrain",
        title="Terrain-Features",
        tile_id=tile_id,
        quiet=quiet,
        fn=lambda p: pipeline.run_terrain(tile_id, str(config) if config else None, progress=p),
    )
    if quiet:
        console.print(f"terrain OK → {result['feature_count']} features")


@app.command("snow-surface")
def snow_surface_cmd(
    tile_id: str = typer.Option(..., help="Tile identifier"),
    profile: str = typer.Option("davos", help="Rendering profile name"),
    config: Optional[Path] = typer.Option(None, help="Path to default.yaml"),
    snow_height: Optional[float] = typer.Option(
        None,
        "--snow-height",
        help="Override base snow height in meters (e.g. 2.0)",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
) -> None:
    """Compute snow-covered surface DEM and thickness raster."""
    result = _run_step(
        step_key="snow_surface",
        title="Schneeoberfläche",
        tile_id=tile_id,
        quiet=quiet,
        profile=profile,
        fn=lambda p: pipeline.run_snow_surface(
            tile_id,
            profile,
            str(config) if config else None,
            snow_height_m=snow_height,
            progress=p,
        ),
    )
    if quiet:
        console.print(f"snow-surface OK → {result['layer_count']} layers")


@app.command("snow")
def snow_cmd(
    tile_id: str = typer.Option(..., help="Tile identifier"),
    profile: str = typer.Option("davos", help="Rendering profile name"),
    config: Optional[Path] = typer.Option(None, help="Path to default.yaml"),
    snow_height: Optional[float] = typer.Option(
        None,
        "--snow-height",
        help="Override base snow height in meters (e.g. 2.0)",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
) -> None:
    """Compute snow coverage intermediate layers."""
    result = _run_step(
        step_key="snow",
        title="Schneebedeckungsmodell",
        tile_id=tile_id,
        quiet=quiet,
        profile=profile,
        fn=lambda p: pipeline.run_snow(
            tile_id,
            profile,
            str(config) if config else None,
            snow_height_m=snow_height,
            progress=p,
        ),
    )
    if quiet:
        console.print(f"snow OK → {result['layer_count']} layers")


@app.command()
def render(
    tile_id: str = typer.Option(..., help="Tile identifier"),
    profile: str = typer.Option("davos", help="Rendering profile name"),
    config: Optional[Path] = typer.Option(None, help="Path to default.yaml"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
) -> None:
    """Render synthetic winter orthophoto."""
    result = _run_step(
        step_key="render",
        title="Winter-Rendering",
        tile_id=tile_id,
        quiet=quiet,
        profile=profile,
        fn=lambda p: pipeline.run_render(
            tile_id, profile, str(config) if config else None, progress=p
        ),
    )
    if quiet:
        console.print(f"render OK → {result['output']}")


@app.command()
def qa(
    tile_id: str = typer.Option(..., help="Tile identifier"),
    config: Optional[Path] = typer.Option(None, help="Path to default.yaml"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
) -> None:
    """Run geometry and plausibility QA checks."""
    result = _run_step(
        step_key="qa",
        title="Qualitätskontrolle",
        tile_id=tile_id,
        quiet=quiet,
        fn=lambda p: pipeline.run_qa_step(tile_id, str(config) if config else None, progress=p),
    )
    status = "PASS" if result["overall_pass"] else "FAIL"
    color = "green" if result["overall_pass"] else "red"
    if not quiet:
        console.print(f"\n[{color}]QA {status}[/{color}]")
    else:
        console.print(f"qa {status}")


@app.command("prepare-region")
def prepare_region_cmd(
    name: str = typer.Option(..., "--name", "-n", help="Region/profile name"),
    extent: str = typer.Option(
        ...,
        "--extent",
        "-e",
        help="Bounding box minx,miny,maxx,maxy in EPSG:2056",
    ),
    config: Optional[Path] = typer.Option(
        None,
        help="Base config to copy (default: config/default.yaml)",
    ),
    tlm_source: Optional[Path] = typer.Option(
        None,
        help="Master swissTLM3D GeoPackage",
    ),
    dem_year: int = typer.Option(
        2023,
        help="Preferred swissALTI3D vintage (falls back per tile via STAC)",
    ),
    wmts_zoom: Optional[int] = typer.Option(
        None,
        help="Override WMTS zoom (default: derived from resolution_m in config)",
    ),
    skip_ortho: bool = typer.Option(False, help="Skip orthophoto download"),
    skip_dem: bool = typer.Option(False, help="Skip DEM download"),
    skip_tlm: bool = typer.Option(False, help="Skip TLM3D extraction"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
) -> None:
    """Download orthophoto/DEM and extract vectors for a region extent."""
    progress = _progress(quiet)

    def report(message: str) -> None:
        if not quiet:
            progress.info(message)

    progress.header("Regionsdaten vorbereiten", tile_id=f"{name}_001", profile=name)
    progress.step_begin(1, 1, "Orthofoto, DEM und Vektoren laden")

    result = prepare_region(
        name=name,
        extent=extent,
        base_config=str(config) if config else None,
        tlm_source=str(tlm_source) if tlm_source else None,
        dem_year=dem_year,
        wmts_zoom=wmts_zoom,
        skip_ortho=skip_ortho,
        skip_dem=skip_dem,
        skip_tlm=skip_tlm,
        progress=report,
    )

    progress.step_end(
        "Regionsdaten vorbereiten",
        f"Config → {result['config']}",
    )
    summary_rows = [
        ("Name", name),
        ("Tile", str(result["tile_id"])),
        ("Profil", name),
        ("Config", str(result["config"])),
        ("Daten", str(result["region_dir"])),
    ]
    ortho = result.get("orthophoto")
    if isinstance(ortho, dict):
        summary_rows.extend(
            [
                (
                    "WMTS Zoom",
                    f"{ortho['zoom']} ({ortho['native_resolution_m']:.2f} m nativ → "
                    f"{ortho['target_resolution_m']:.2f} m)",
                ),
                ("WMTS Kacheln", str(ortho["tile_count"])),
                (
                    "WMTS col/row",
                    f"{ortho['col_range'][0]}–{ortho['col_range'][1]} / "
                    f"{ortho['row_range'][0]}–{ortho['row_range'][1]}",
                ),
            ]
        )
    progress.summary_table(summary_rows)
    console.print(f"\n[bold green]Bereit:[/bold green] {result['run_command']}")


@app.command("extract-tlm3d")
def extract_tlm3d_cmd(
    tile_id: str = typer.Option("davos_001", help="Tile id with bbox in config"),
    source: Optional[Path] = typer.Option(
        None,
        help="Master swissTLM3D GeoPackage",
    ),
    config: Optional[Path] = typer.Option(None, help="Path to default.yaml"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
) -> None:
    """Extract clipped TLM3D layers from the national GeoPackage."""
    progress = _progress(quiet)
    progress.header("TLM3D-Extraktion", tile_id=tile_id)
    progress.step_begin(1, 1, "Layer aus swissTLM3D extrahieren")

    script = Path(__file__).resolve().parents[1] / "scripts" / "extract_tlm3d.py"
    cmd = [sys.executable, str(script), "--tile-id", tile_id]
    if config:
        cmd.extend(["--config", str(config)])
    if source:
        cmd.extend(["--source", str(source)])

    subprocess.check_call(cmd)
    progress.step_end("TLM3D-Extraktion", f"7 Layer → data/raw/tlm3d/")


@app.command("run-all")
def run_all_cmd(
    tile_id: str = typer.Option(..., help="Tile identifier"),
    profile: str = typer.Option("davos", help="Rendering profile name"),
    config: Optional[Path] = typer.Option(None, help="Path to default.yaml"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
    json_output: bool = typer.Option(False, "--json", help="Print full result JSON at end"),
) -> None:
    """Run the full pipeline end-to-end."""
    progress = _progress(quiet)
    progress.header("Winter-Orthofoto Pipeline", tile_id=tile_id, profile=profile)

    if not quiet:
        progress.info(
            "Schritte: " + " → ".join(title for _, title, _ in PIPELINE_STEPS)
        )

    step_names = [title for _, title, _ in PIPELINE_STEPS]

    if quiet:
        result = pipeline.run_all(
            tile_id, profile, str(config) if config else None, progress=None
        )
        for name in step_names:
            console.print(f"  ✓ {name}")
    else:
        with progress.track_steps(step_names) as tracked:
            result = pipeline.run_all(
                tile_id, profile, str(config) if config else None, progress=tracked
            )

    qa_pass = result["qa"]["overall_pass"]
    color = "green" if qa_pass else "yellow"
    status = "PASS" if qa_pass else "FAIL"

    console.print(Rule(f"[bold {color}]Pipeline abgeschlossen — QA {status}[/bold {color}]"))
    progress.summary_table(
        [
            ("Tile", tile_id),
            ("Profil", profile),
            ("Winter-RGB", str(result["render"]["output"])),
            ("QA", status),
        ]
    )

    if json_output:
        console.print_json(json.dumps(result, default=str))

    _export_viewer_after_pipeline(tile_id, config, quiet=quiet)


@app.command("run-all-snow")
def run_all_snow_cmd(
    tile_id: str = typer.Option(..., help="Tile identifier"),
    profile: str = typer.Option("davos", help="Rendering profile name"),
    config: Optional[Path] = typer.Option(None, help="Path to default.yaml"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
    json_output: bool = typer.Option(False, "--json", help="Print full result JSON at end"),
) -> None:
    """Run pipeline from snow-surface onwards (requires prior harmonize/masks/terrain)."""
    progress = _progress(quiet)
    progress.header("Winter-Orthofoto Pipeline (ab Schneeoberfläche)", tile_id=tile_id, profile=profile)

    step_names = [title for _, title, _ in pipeline.PIPELINE_STEPS_SNOW]

    if not quiet:
        progress.info("Schritte: " + " → ".join(step_names))

    if quiet:
        result = pipeline.run_all_snow(
            tile_id, profile, str(config) if config else None, progress=None
        )
        for name in step_names:
            console.print(f"  ✓ {name}")
    else:
        with progress.track_steps(step_names) as tracked:
            result = pipeline.run_all_snow(
                tile_id, profile, str(config) if config else None, progress=tracked
            )

    qa_pass = result["qa"]["overall_pass"]
    color = "green" if qa_pass else "yellow"
    status = "PASS" if qa_pass else "FAIL"

    console.print(Rule(f"[bold {color}]Pipeline abgeschlossen — QA {status}[/bold {color}]"))
    progress.summary_table(
        [
            ("Tile", tile_id),
            ("Profil", profile),
            ("Winter-RGB", str(result["render"]["output"])),
            ("QA", status),
        ]
    )

    if json_output:
        console.print_json(json.dumps(result, default=str))

    _export_viewer_after_pipeline(tile_id, config, quiet=quiet)


@app.command("viewer-export")
def viewer_export_cmd(
    tile_id: str = typer.Option(..., help="Tile identifier"),
    config: Optional[Path] = typer.Option(None, help="Path to default.yaml"),
    output: Optional[Path] = typer.Option(
        None,
        help="Output directory (default: viewer/data/{tile_id})",
    ),
    stride: Optional[int] = typer.Option(
        VIEWER_EXPORT_STRIDE,
        help=f"Mesh decimation stride (default: {VIEWER_EXPORT_STRIDE}, same as run-all)",
    ),
    max_texture_dim: int = typer.Option(
        VIEWER_EXPORT_MAX_TEXTURE_DIM,
        help=f"Max orthophoto texture edge length in pixels (default: {VIEWER_EXPORT_MAX_TEXTURE_DIM})",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
) -> None:
    """Export DEM mesh and ortho textures for the 3D viewer."""
    result = export_tile_viewer_data(
        tile_id,
        config_path=str(config) if config else None,
        output_dir=output,
        stride=stride,
        max_texture_dim=max_texture_dim,
    )
    if quiet:
        console.print(f"viewer-export OK → {result['output_dir']}")
    else:
        console.print(
            f"[green]Exportiert:[/green] {result['output_dir']}\n"
            f"  {result['vertex_count']:,} Vertices, "
            f"{result['triangle_count']:,} Dreiecke (stride={result['stride']})\n"
            f"  Textur {result['texture_width']}×{result['texture_height']} px "
            f"(stride={result['texture_stride']}, max={result['max_texture_dim']})"
        )


@app.command("viewer")
def viewer_cmd(
    tile_id: str = typer.Option(..., help="Tile identifier"),
    config: Optional[Path] = typer.Option(None, help="Path to default.yaml"),
    host: str = typer.Option(
        "0.0.0.0",
        help="Bind address (0.0.0.0 = all interfaces, 127.0.0.1 = localhost only)",
    ),
    port: int = typer.Option(8765, help="HTTP port for the viewer"),
    stride: Optional[int] = typer.Option(None, help="Mesh decimation stride"),
    max_texture_dim: int = typer.Option(
        1024,
        help="Max orthophoto texture edge length in pixels",
    ),
    re_export: bool = typer.Option(
        False,
        "--re-export",
        help="Re-run viewer-export even if data already exists",
    ),
    no_browser: bool = typer.Option(False, help="Do not open a browser tab"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
) -> None:
    """Start the 3D viewer (exports tile data only when missing or --re-export)."""
    data_dir = get_project_root() / "viewer" / "data" / tile_id
    if re_export or not (data_dir / "scene.json").exists():
        result = export_tile_viewer_data(
            tile_id,
            config_path=str(config) if config else None,
            stride=stride,
            max_texture_dim=max_texture_dim,
        )
        if not quiet:
            console.print(
                f"[green]Exportiert:[/green] {result['output_dir']}\n"
                f"  {result['vertex_count']:,} Vertices, "
                f"{result['triangle_count']:,} Dreiecke (stride={result['stride']})\n"
                f"  Textur {result['texture_width']}×{result['texture_height']} px "
                f"(stride={result['texture_stride']}, max={result['max_texture_dim']})"
            )
    elif not quiet:
        console.print(
            f"[dim]Nutze vorhandenen Export:[/dim] {data_dir}\n"
            "[dim](--re-export zum Neu-Exportieren mit anderen Qualitätsparametern)[/dim]"
        )

    viewer_dir = get_project_root() / "viewer"
    local_url = f"http://127.0.0.1:{port}/?tile={tile_id}"

    if not quiet:
        console.print(f"[bold]Viewer:[/bold] {local_url}")
        if host in ("0.0.0.0", "::"):
            console.print(
                f"[dim]Erreichbar im Netzwerk unter http://<host-ip>:{port}/?tile={tile_id}[/dim]"
            )

    handler = partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(viewer_dir),
    )
    server = http.server.ThreadingHTTPServer((host, port), handler)

    if not no_browser:
        webbrowser.open(local_url)

    if not quiet:
        console.print("Beenden mit Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if not quiet:
            console.print("\nViewer beendet.")


if __name__ == "__main__":
    app()
