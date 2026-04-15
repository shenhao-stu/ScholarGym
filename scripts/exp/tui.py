#!/usr/bin/env python3
"""ScholarGym experiment TUI (textual-based, full-featured).

Two-pane layout:
    Left  — table of all runs grouped by type -> group
    Right — top status/metric panel + bottom scrollable log panel

Key bindings (shown in the footer):
    Tab          switch focus between run table and log pane
    j / ↓        next row (when table focused) / scroll down (when log focused)
    k / ↑        previous row (when table focused) / scroll up (when log focused)
    PageDown/Up  scroll log pane
    End / Home   jump log pane to latest / top
    r            refresh now
    K            kill selected (with confirmation)
    R            restart selected (with confirmation)
    F            restart selected with --fresh (wipes checkpoint, with confirmation)
    L            toggle right panel visibility
    q / Ctrl+C   quit

Usage:
    python scripts/exp/tui.py
    python scripts/exp/tui.py --runs-dir runs --interval 5
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    Static,
)

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from exp.state import (  # noqa: E402
    RunSnapshot,
    extract_metric_history,
    fmt_duration,
    list_run_dirs,
    read_log_tail_lines,
    read_snapshot,
    sparkline,
    _tail_log,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


STATUS_STYLE = {
    "running": "bold green",
    "done": "bold blue",
    "crashed": "bold red",
    "stalled": "bold yellow",
    "stopped": "dim white",
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


# ---------------------------------------------------------------------------
# Modal: confirm destructive action
# ---------------------------------------------------------------------------

class ConfirmScreen(ModalScreen[bool]):
    """Yes/No confirmation modal."""

    BINDINGS = [
        Binding("y", "confirm(True)", "Yes"),
        Binding("n", "confirm(False)", "No"),
        Binding("escape", "confirm(False)", "Cancel"),
    ]

    DEFAULT_CSS = """
    ConfirmScreen {
        align: center middle;
    }
    #confirm-box {
        width: 72;
        min-height: 9;
        height: auto;
        border: thick $accent;
        background: $panel;
        padding: 1 2;
    }
    #confirm-msg {
        width: 1fr;
        content-align: center middle;
        text-align: center;
        margin: 1 0;
    }
    #confirm-hint {
        width: 1fr;
        content-align: center middle;
        text-align: center;
        color: $text-muted;
    }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with Container(id="confirm-box"):
            yield Label(self.message, id="confirm-msg")
            yield Label("y = yes   n = no   esc = cancel", id="confirm-hint")

    def action_confirm(self, ok: bool) -> None:
        self.dismiss(ok)


# ---------------------------------------------------------------------------
# Right-side panels
# ---------------------------------------------------------------------------

class DetailPanel(Static):
    """Shows selected run status and metric summary."""

    DEFAULT_CSS = """
    DetailPanel {
        padding: 1 2;
        background: $surface;
    }
    """


class LogPanel(Static):
    """Shows a scrollable log tail."""

    DEFAULT_CSS = """
    LogPanel {
        padding: 1 2;
        background: $surface;
    }
    """


class LogScroll(VerticalScroll):
    """Scrollable log pane with local key bindings."""

    class FocusLog(Message):
        pass

    BINDINGS = [
        Binding("j", "scroll_down", "Down", show=False),
        Binding("k", "scroll_up", "Up", show=False),
        Binding("down", "scroll_down", show=False),
        Binding("up", "scroll_up", show=False),
        Binding("pagedown", "page_down", show=False),
        Binding("pageup", "page_up", show=False),
        Binding("end", "to_end", show=False),
        Binding("home", "to_home", show=False),
    ]

    def on_focus(self) -> None:
        self.post_message(self.FocusLog())

    def action_scroll_down(self) -> None:
        self.scroll_relative(y=3, animate=False)

    def action_scroll_up(self) -> None:
        self.scroll_relative(y=-3, animate=False)

    def action_page_down(self) -> None:
        self.scroll_relative(y=10, animate=False)

    def action_page_up(self) -> None:
        self.scroll_relative(y=-10, animate=False)

    def action_to_end(self) -> None:
        self.scroll_end(animate=False)

    def action_to_home(self) -> None:
        self.scroll_home(animate=False)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

class ExperimentTUI(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #main {
        height: 1fr;
        layout: horizontal;
    }
    #table-pane {
        width: 60%;
        border-right: solid $accent;
    }
    #detail-pane {
        width: 40%;
        layout: vertical;
    }
    #detail-top {
        height: 16;
        border-bottom: solid $accent;
    }
    #log-scroll {
        height: 1fr;
    }
    DataTable {
        height: 100%;
    }
    DetailPanel {
        height: auto;
    }
    LogPanel {
        height: auto;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("tab", "toggle_focus", "Focus"),
        Binding("r", "refresh_now", "Refresh"),
        Binding("j", "context_down", "↓", show=False),
        Binding("k", "context_up", "↑", show=False),
        Binding("down", "context_down", show=False),
        Binding("up", "context_up", show=False),
        Binding("pagedown", "log_page_down", "Log↓"),
        Binding("pageup", "log_page_up", "Log↑"),
        Binding("end", "log_end", "Latest"),
        Binding("home", "log_home", "Top"),
        Binding("K", "kill_selected", "Kill"),
        Binding("R", "restart_selected", "Restart"),
        Binding("F", "restart_fresh", "Restart-Fresh"),
        Binding("L", "toggle_detail", "Toggle Panel"),
    ]

    def __init__(self, runs_dir: Path, interval: float, manifest: Path) -> None:
        super().__init__()
        self.runs_dir = runs_dir
        self.interval = interval
        self.manifest = manifest
        self.snaps: list[RunSnapshot] = []
        # Map row key (as string) → snapshot index
        self.row_to_snap: dict[str, int] = {}
        self.detail_visible = True
        self._last_selected_run: Optional[str] = None
        self.focus_on_log = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="table-pane"):
                yield DataTable(id="run-table", cursor_type="row", zebra_stripes=True)
            with Vertical(id="detail-pane"):
                with Vertical(id="detail-top"):
                    yield DetailPanel("Select a run on the left.", id="detail")
                with LogScroll(id="log-scroll"):
                    yield LogPanel("(no log yet)", id="log")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#run-table", DataTable)
        table.add_columns("Name", "Type", "Status", "Progress", "Done", "Elapsed", "ETA", "R/P")
        table.focus()
        self.refresh_data()
        self.set_interval(self.interval, self.refresh_data)
        self.title = f"ScholarGym TUI — {self.runs_dir}"

    # ---------------------- data refresh ----------------------

    def refresh_data(self) -> None:
        """Re-read all run dirs and rebuild the table. Preserve cursor."""
        self.snaps = [read_snapshot(d) for d in list_run_dirs(self.runs_dir)]
        # Sort: by type then group then name
        self.snaps.sort(key=lambda s: (s.exp_type, s.group, s.name))

        table = self.query_one("#run-table", DataTable)
        # Save cursor
        prev_key: Optional[str] = None
        if table.row_count and table.cursor_row is not None and 0 <= table.cursor_row < table.row_count:
            try:
                prev_key = table.coordinate_to_cell_key((table.cursor_row, 0)).row_key.value
            except Exception:
                prev_key = None

        table.clear(columns=False)
        self.row_to_snap.clear()

        # Insert grouped rows: type -> group -> run
        last_type = None
        last_group = None
        for i, s in enumerate(self.snaps):
            if s.exp_type != last_type:
                table.add_row(f"▼ {s.exp_type}", "", "", "", "", "", "", "", key=f"_type_{s.exp_type}")
                last_type = s.exp_type
                last_group = None
            if s.group != last_group:
                table.add_row(f"  ▸ {s.group}", "", "", "", "", "", "", "", key=f"_grp_{s.exp_type}_{s.group}")
                last_group = s.group
            row_key = f"snap_{i}"
            self.row_to_snap[row_key] = i

            style = STATUS_STYLE.get(s.status, "white")
            glyph = STATUS_GLYPH.get(s.status, "?")
            from rich.text import Text  # local import

            status_cell = Text(f"{glyph} {s.status}", style=style)
            bar_w = 12
            filled = int(round(s.progress_ratio * bar_w))
            bar = "█" * filled + "░" * (bar_w - filled)
            progress_cell = f"{bar} {s.progress_ratio*100:5.1f}%"
            done_cell = f"{s.done_queries}/{s.total_queries}"
            elapsed_cell = fmt_duration(s.elapsed_sec) if s.elapsed_sec > 0 else "--"
            eta_cell = fmt_duration(s.eta_sec)
            metric_cell = ""
            if s.last_metric:
                m = s.last_metric
                metric_cell = f"R={m['sel_r']:.2f} P={m['sel_p']:.2f}"
            elif s.anomaly:
                metric_cell = Text(f"⚠ {s.anomaly}", style="bold red")
            table.add_row(
                f"  {s.name}",
                s.exp_type,
                status_cell,
                progress_cell,
                done_cell,
                elapsed_cell,
                eta_cell,
                metric_cell,
                key=row_key,
            )

        # Try to restore cursor
        if prev_key:
            for r in range(table.row_count):
                try:
                    if table.coordinate_to_cell_key((r, 0)).row_key.value == prev_key:
                        table.move_cursor(row=r)
                        break
                except Exception:
                    pass

        self.update_detail()

    # ---------------------- selection / detail ----------------------

    def selected_snapshot(self) -> Optional[RunSnapshot]:
        table = self.query_one("#run-table", DataTable)
        if table.cursor_row is None or table.cursor_row >= table.row_count:
            return None
        try:
            key = table.coordinate_to_cell_key((table.cursor_row, 0)).row_key.value
        except Exception:
            return None
        if not key or key.startswith("_grp_") or key.startswith("_type_"):
            return None
        idx = self.row_to_snap.get(key)
        if idx is None:
            return None
        return self.snaps[idx]

    def update_detail(self) -> None:
        panel = self.query_one("#detail", DetailPanel)
        log_panel = self.query_one("#log", LogPanel)
        s = self.selected_snapshot()
        if s is None:
            self._last_selected_run = None
            panel.update("No selection. Use Tab to switch focus and j/k to navigate.")
            log_panel.update("(no log yet)")
            return

        should_follow_latest = self._last_selected_run != s.name
        self._last_selected_run = s.name

        # Pull tail / history
        tail = _tail_log(s.run_dir)
        history = extract_metric_history(tail)
        sel_r = [h["sel_r"] for h in history]
        sel_p = [h["sel_p"] for h in history]
        ret_r = [h["ret_r"] for h in history]

        log_lines = read_log_tail_lines(s.run_dir, n_lines=120)
        log_text = "\n".join(log_lines) if log_lines else "(no log yet)"

        sl_sel_r = sparkline(sel_r, width=30) if sel_r else " " * 30
        sl_sel_p = sparkline(sel_p, width=30) if sel_p else " " * 30
        sl_ret_r = sparkline(ret_r, width=30) if ret_r else " " * 30

        anomaly_line = f"\n[bold red]⚠ anomaly: {s.anomaly}[/bold red]" if s.anomaly else ""
        last_metric_line = ""
        if s.last_metric:
            m = s.last_metric
            last_metric_line = (
                f"\nLast iter {m['iter']}: "
                f"sel R={m['sel_r']:.3f} P={m['sel_p']:.3f}  "
                f"ret R={m['ret_r']:.3f} P={m['ret_p']:.3f}"
            )

        body = (
            f"[bold]{s.name}[/bold]   group=[cyan]{s.group}[/cyan]   type=[magenta]{s.exp_type}[/magenta]\n"
            f"status: [{STATUS_STYLE.get(s.status, 'white')}]{s.status}[/]"
            f"   pid={s.pid or '-'}\n"
            f"start: {s.start_time.isoformat(timespec='seconds') if s.start_time else '-'}\n"
            f"progress: {s.done_queries}/{s.total_queries} ({s.progress_ratio*100:.1f}%)\n"
            f"elapsed: {fmt_duration(s.elapsed_sec)}    eta: {fmt_duration(s.eta_sec)}"
            f"{last_metric_line}{anomaly_line}\n"
            f"\n[bold]Metric history[/bold]   ({len(history)} points)\n"
            f"  sel R  [green]{sl_sel_r}[/green]\n"
            f"  sel P  [yellow]{sl_sel_p}[/yellow]\n"
            f"  ret R  [cyan]{sl_ret_r}[/cyan]\n"
            f"\n[dim]Tab switches focus between table and log. j/k scroll the focused pane.[/dim]"
        )
        panel.update(body)
        log_panel.update(log_text)
        if should_follow_latest:
            self.call_after_refresh(self._scroll_detail_to_latest)

    # ---------------------- actions ----------------------

    def action_refresh_now(self) -> None:
        self.refresh_data()

    def action_toggle_focus(self) -> None:
        self.focus_on_log = not self.focus_on_log
        if self.focus_on_log:
            self._log_scroll().focus()
            self.notify("focus: log", severity="information")
        else:
            self.query_one("#run-table", DataTable).focus()
            self.notify("focus: table", severity="information")

    def on_log_scroll_focus_log(self, _: LogScroll.FocusLog) -> None:
        self.focus_on_log = True

    def on_data_table_focus(self) -> None:
        self.focus_on_log = False

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        table = self.query_one("#run-table", DataTable)
        try:
            key = table.coordinate_to_cell_key((event.cursor_row, 0)).row_key.value
        except Exception:
            key = None
        if key and not key.startswith("_grp_") and not key.startswith("_type_"):
            self._last_selected_run = None
        self.update_detail()

    def action_context_down(self) -> None:
        if self.focus_on_log:
            self._log_scroll().scroll_relative(y=3, animate=False)
        else:
            self.query_one("#run-table", DataTable).action_cursor_down()

    def action_context_up(self) -> None:
        if self.focus_on_log:
            self._log_scroll().scroll_relative(y=-3, animate=False)
        else:
            self.query_one("#run-table", DataTable).action_cursor_up()

    def action_toggle_detail(self) -> None:
        pane = self.query_one("#detail-pane")
        table_pane = self.query_one("#table-pane")
        self.detail_visible = not self.detail_visible
        if self.detail_visible:
            pane.styles.display = "block"
            table_pane.styles.width = "60%"
            self.call_after_refresh(self._scroll_detail_to_latest)
        else:
            pane.styles.display = "none"
            table_pane.styles.width = "100%"

    def _log_scroll(self) -> LogScroll:
        return self.query_one("#log-scroll", LogScroll)

    def _scroll_detail_to_latest(self) -> None:
        self._log_scroll().scroll_end(animate=False)

    def action_log_page_down(self) -> None:
        self._log_scroll().scroll_relative(y=10, animate=False)

    def action_log_page_up(self) -> None:
        self._log_scroll().scroll_relative(y=-10, animate=False)

    def action_log_end(self) -> None:
        self._scroll_detail_to_latest()

    def action_log_home(self) -> None:
        self._log_scroll().scroll_home(animate=False)

    def _run_launcher(self, *args: str) -> None:
        """Invoke launcher.py as a subprocess. Output goes to stderr (visible after exit)."""
        cmd = [sys.executable, "scripts/exp/launcher.py", "--manifest", str(self.manifest), *args]
        try:
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
        except Exception as e:
            self.notify(f"launcher failed: {e}", severity="error")

    def _confirm(self, message: str, on_confirm) -> None:
        self.push_screen(ConfirmScreen(message), callback=on_confirm)

    def _after_kill_confirm(self, ok: bool) -> None:
        if not ok:
            return
        s = self.selected_snapshot()
        if s is None:
            return
        self._run_launcher("down", "--only", s.name)
        self.notify(f"sent SIGTERM to {s.name}", severity="information")
        self.refresh_data()

    def _after_restart_confirm(self, ok: bool) -> None:
        if not ok:
            return
        s = self.selected_snapshot()
        if s is None:
            return
        self._run_launcher("restart", "--only", s.name)
        self.notify(f"restarted {s.name}", severity="information")
        self.refresh_data()

    def _after_restart_fresh_confirm(self, ok: bool) -> None:
        if not ok:
            return
        s = self.selected_snapshot()
        if s is None:
            return
        self._run_launcher("restart", "--only", s.name, "--fresh")
        self.notify(f"fresh-restarted {s.name}", severity="information")
        self.refresh_data()

    def action_kill_selected(self) -> None:
        s = self.selected_snapshot()
        if s is None:
            self.notify("no selection", severity="warning")
            return
        if s.status not in ("running", "stalled"):
            self.notify(f"{s.name} is not running", severity="warning")
            return
        self._confirm(f"Kill experiment '{s.name}' (pid={s.pid})?", self._after_kill_confirm)

    def action_restart_selected(self) -> None:
        s = self.selected_snapshot()
        if s is None:
            self.notify("no selection", severity="warning")
            return
        self._confirm(
            f"Restart '{s.name}'? (down + up, checkpoint preserved)",
            self._after_restart_confirm,
        )

    def action_restart_fresh(self) -> None:
        s = self.selected_snapshot()
        if s is None:
            self.notify("no selection", severity="warning")
            return
        self._confirm(
            f"FRESH restart '{s.name}'?\nThis WIPES the run dir and checkpoint!",
            self._after_restart_fresh_confirm,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs", help="Runs directory")
    parser.add_argument("--manifest", default="experiments.yaml", help="Manifest path (for kill/restart)")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval seconds")
    args = parser.parse_args()

    runs_root = (PROJECT_ROOT / args.runs_dir).resolve()
    manifest = (PROJECT_ROOT / args.manifest).resolve()

    app = ExperimentTUI(runs_dir=runs_root, interval=args.interval, manifest=manifest)
    app.run()


if __name__ == "__main__":
    main()
