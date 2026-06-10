from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table


class PipelineProgress:
    """Rich-based progress reporting for pipeline steps."""

    def __init__(self, *, console: Console | None = None, verbose: bool = True) -> None:
        self.console = console or Console()
        self.verbose = verbose
        self._step_start: float | None = None

    def header(self, title: str, *, tile_id: str | None = None, profile: str | None = None) -> None:
        if not self.verbose:
            return
        parts = [f"[bold cyan]{title}[/bold cyan]"]
        if tile_id:
            parts.append(f"tile=[yellow]{tile_id}[/yellow]")
        if profile:
            parts.append(f"profile=[yellow]{profile}[/yellow]")
        self.console.print(Rule(" · ".join(parts)))

    def step_begin(self, index: int, total: int, name: str, detail: str = "") -> None:
        if not self.verbose:
            return
        self._step_start = time.perf_counter()
        label = f"[bold][{index}/{total}][/bold] {name}"
        if detail:
            label += f" — [dim]{detail}[/dim]"
        self.console.print(label)

    def step_end(self, name: str, summary: str = "") -> None:
        if not self.verbose:
            return
        elapsed = ""
        if self._step_start is not None:
            elapsed = f" [dim]({time.perf_counter() - self._step_start:.1f}s)[/dim]"
        msg = f"  [green]✓[/green] {name}{elapsed}"
        if summary:
            msg += f" — {summary}"
        self.console.print(msg)
        self._step_start = None

    def substep(self, message: str) -> None:
        if not self.verbose:
            return
        self.console.print(f"  [dim]→[/dim] {message}")

    def warn(self, message: str) -> None:
        if not self.verbose:
            return
        self.console.print(f"  [yellow]![/yellow] {message}")

    def info(self, message: str) -> None:
        if not self.verbose:
            return
        self.console.print(f"  [blue]i[/blue] {message}")

    def summary_table(self, rows: list[tuple[str, str]]) -> None:
        if not self.verbose:
            return
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("key", style="dim")
        table.add_column("value")
        for key, value in rows:
            table.add_row(key, value)
        self.console.print(table)

    @contextmanager
    def track_steps(self, steps: list[str]) -> Iterator[PipelineProgress]:
        if not self.verbose:
            yield self
            return
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        ) as progress:
            task_id = progress.add_task("Pipeline", total=len(steps))
            self._progress = progress
            self._task_id = task_id
            self._steps = steps
            self._index = 0
            yield self
            progress.update(task_id, completed=len(steps))

    def advance(self, step_name: str) -> None:
        if not self.verbose or not hasattr(self, "_progress"):
            return
        self._index += 1
        self._progress.update(
            self._task_id,
            advance=1,
            description=f"[{self._index}/{len(self._steps)}] {step_name}",
        )
