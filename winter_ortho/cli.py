from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.rule import Rule

from winter_ortho import pipeline
from winter_ortho.pipeline import PIPELINE_STEPS
from winter_ortho.utils.progress import PipelineProgress

app = typer.Typer(help="Winter orthophoto generation pipeline")
console = Console()


def _progress(quiet: bool) -> PipelineProgress:
    return PipelineProgress(console=console, verbose=not quiet)


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


@app.command("snow")
def snow_cmd(
    tile_id: str = typer.Option(..., help="Tile identifier"),
    profile: str = typer.Option("davos", help="Rendering profile name"),
    config: Optional[Path] = typer.Option(None, help="Path to default.yaml"),
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
            tile_id, profile, str(config) if config else None, progress=p
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


if __name__ == "__main__":
    app()
