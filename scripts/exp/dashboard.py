#!/usr/bin/env python3
"""ScholarGym experiment dashboard.

A live TUI built on rich.live that polls runs/ every few seconds, derives
per-run progress / ETA / anomaly state, and renders a grouped table.

It does NOT touch the experiment processes — purely a viewer. Ctrl+C exits.

Usage:
    python scripts/exp/dashboard.py                  # default: runs/, 5s refresh
    python scripts/exp/dashboard.py --runs-dir runs --interval 3
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from exp.state import (  # noqa: E402
    RunSnapshot,
    fmt_duration,
    list_run_dirs,
    read_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Status → display style mapping
STATUS_STYLE = {
    "running": "bold green",
    "done": "bold blue",
    "crashed": "bold red on white",
    "stalled": "bold yellow",
    "stopped": "dim",
    "unknown": "dim",
}

STATUS_GLYPH = {
    "running": "▶",
    "done": "✓",
    "crashed": "✗",
    "stalled": "⧗",
    "stopped": "⊘",
    "unknown": "?",
}


def progress_bar(ratio: float, width: int = 14) -> str:
    filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


def render_table(runs_root: Path, snaps: list[RunSnapshot]) -> Table:
    table = Table(
        title=f"ScholarGym Experiments — {runs_root}   {datetime.now():%Y-%m-%d %H:%M:%S}",
        title_style="bold",
        header_style="bold cyan",
        expand=True,
        pad_edge=False,
    )
    table.add_column("Name", style="white", no_wrap=True)
    table.add_column("Type", justify="center", width=10)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Progress", width=18)
    table.add_column("Done", justify="right", width=10)
    table.add_column("Elapsed", justify="right", width=8)
    table.add_column("ETA", justify="right", width=8)
    table.add_column("Last Metric", justify="left", width=20)
    table.add_column("Anomaly", justify="left", overflow="fold")

    if not snaps:
        table.add_row("(no runs)", "", "", "", "", "", "", "", "")
        return table

    grouped: dict[str, dict[str, list[RunSnapshot]]] = defaultdict(lambda: defaultdict(list))
    for s in snaps:
        grouped[s.exp_type][s.group].append(s)

    first_type = True
    for exp_type in sorted(grouped):
        if not first_type:
            table.add_section()
        first_type = False
        table.add_row(
            Text(f"▼ {exp_type}", style="bold yellow"),
            "", "", "", "", "", "", "", "",
        )
        for group in sorted(grouped[exp_type]):
            table.add_row(
                Text(f"  ▸ {group}", style="bold magenta"),
                "", "", "", "", "", "", "", "",
            )
            for s in sorted(grouped[exp_type][group], key=lambda r: r.name):
                style = STATUS_STYLE.get(s.status, "white")
                glyph = STATUS_GLYPH.get(s.status, "?")
                status_cell = Text(f"{glyph} {s.status}", style=style)

                bar = progress_bar(s.progress_ratio)
                bar_text = Text(f"{bar} {s.progress_ratio*100:5.1f}%")

                done_cell = f"{s.done_queries}/{s.total_queries}"
                elapsed_cell = fmt_duration(s.elapsed_sec) if s.elapsed_sec > 0 else "--"
                eta_cell = fmt_duration(s.eta_sec)

                if s.last_metric:
                    m = s.last_metric
                    metric_cell = (
                        f"i{m['iter']} R={m['sel_r']:.2f} P={m['sel_p']:.2f}"
                    )
                else:
                    metric_cell = "—"

                anomaly_cell = ""
                if s.anomaly:
                    anomaly_cell = Text(f"⚠ {s.anomaly}", style="bold red")
                elif s.status == "crashed":
                    anomaly_cell = Text("process gone", style="bold red")

                table.add_row(
                    f"  {s.name}",
                    s.exp_type,
                    status_cell,
                    bar_text,
                    done_cell,
                    elapsed_cell,
                    eta_cell,
                    metric_cell,
                    anomaly_cell,
                )

    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs", help="Runs directory to monitor")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval seconds")
    parser.add_argument("--once", action="store_true", help="Render once and exit (no live loop)")
    args = parser.parse_args()

    runs_root = (PROJECT_ROOT / args.runs_dir).resolve()
    console = Console()

    if args.once:
        snaps = [read_snapshot(d) for d in list_run_dirs(runs_root)]
        console.print(render_table(runs_root, snaps))
        return

    try:
        with Live(console=console, refresh_per_second=4, screen=False) as live:
            while True:
                snaps = [read_snapshot(d) for d in list_run_dirs(runs_root)]
                live.update(render_table(runs_root, snaps))
                time.sleep(args.interval)
    except KeyboardInterrupt:
        console.print("[dim]dashboard exited (experiments unaffected)[/dim]")


if __name__ == "__main__":
    main()
