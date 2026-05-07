#!/usr/bin/env python3
"""ScholarGym experiment TUI (textual-based, full-featured).

Two-pane layout:
    Left  — tree of all runs: exp_type → model → run leaf
    Right — top status/metric panel + bottom scrollable log panel

Key bindings (shown in the footer):
    Tab          switch focus between run tree and log pane
    j / ↓        next node (when tree focused) / scroll down (when log focused)
    k / ↑        previous node (when tree focused) / scroll up (when log focused)
    Enter        expand/collapse branch (when tree focused)
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
    Footer,
    Header,
    Label,
    Static,
    Tree,
)
from textual.widgets.tree import TreeNode

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from exp.state import (  # noqa: E402
    RunSnapshot,
    extract_metric_history,
    fmt_duration,
    fmt_settings_badge,
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
    Tree {
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
        # Map run name → TreeNode for quick cursor restore across refreshes.
        self._leaf_nodes_by_name: dict[str, TreeNode] = {}
        self.detail_visible = True
        self._last_selected_run: Optional[str] = None
        self.focus_on_log = False
        # After the first refresh, stop auto-expanding type nodes so the user's
        # manual collapse choices are preserved.
        self._first_refresh_done = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="table-pane"):
                tree: Tree[dict] = Tree("runs", id="run-tree")
                tree.show_root = False
                tree.guide_depth = 3
                yield tree
            with Vertical(id="detail-pane"):
                with Vertical(id="detail-top"):
                    yield DetailPanel("Select a run on the left.", id="detail")
                with LogScroll(id="log-scroll"):
                    yield LogPanel("(no log yet)", id="log")
        yield Footer()

    def on_mount(self) -> None:
        tree = self.query_one("#run-tree", Tree)
        tree.focus()
        self.refresh_data()
        self.set_interval(self.interval, self.refresh_data)
        self.title = f"ScholarGym TUI — {self.runs_dir}"

    # ---------------------- data refresh ----------------------

    def _build_leaf_label(self, s: RunSnapshot):
        """Render a run's status line as a Rich Text for a tree leaf."""
        from rich.text import Text  # local import (matches existing pattern)

        settings = s.settings or {}
        search = str(settings.get("search", "bm25"))
        browser = str(settings.get("browser", "NONE")).lower()
        thinking = "think" if settings.get("thinking") else "nothink"
        axis = f"{search}/{browser}/{thinking}"

        glyph = STATUS_GLYPH.get(s.status, "?")
        status_style = STATUS_STYLE.get(s.status, "white")

        bar_w = 10
        filled = int(round(s.progress_ratio * bar_w))
        bar = "█" * filled + "░" * (bar_w - filled)
        pct = f"{s.progress_ratio*100:4.0f}%"
        done_str = f"{s.done_queries}/{s.total_queries}"
        elapsed = fmt_duration(s.elapsed_sec) if s.elapsed_sec > 0 else "--"

        metric_str = ""
        metric_style = "dim"
        if s.last_metric:
            m = s.last_metric
            metric_str = f"R={m['sel_r']:.2f} P={m['sel_p']:.2f}"
        elif s.anomaly:
            metric_str = f"⚠{s.anomaly}"
            metric_style = "bold red"

        text = Text()
        text.append(f"{axis:<26}")
        text.append(f"{glyph} {s.status:<8}", style=status_style)
        text.append(f"  {bar} {pct}")
        text.append(f"  {done_str:>11}  {elapsed:>7}")
        if metric_str:
            text.append(f"  {metric_str}", style=metric_style)
        return text

    def _build_branch_label(self, title: str, snaps: list[RunSnapshot]):
        """Render a type/model branch label with an aggregate status pill."""
        from rich.text import Text

        counts: dict[str, int] = defaultdict(int)
        for s in snaps:
            counts[s.status] += 1

        # Decide an aggregate style: crashed/stalled dominates, then running, then done.
        if counts.get("crashed"):
            pill_glyph = STATUS_GLYPH["crashed"]
            pill_style = STATUS_STYLE["crashed"]
        elif counts.get("stalled"):
            pill_glyph = STATUS_GLYPH["stalled"]
            pill_style = STATUS_STYLE["stalled"]
        elif counts.get("running"):
            pill_glyph = STATUS_GLYPH["running"]
            pill_style = STATUS_STYLE["running"]
        elif counts.get("done") and counts["done"] == len(snaps):
            pill_glyph = STATUS_GLYPH["done"]
            pill_style = STATUS_STYLE["done"]
        else:
            pill_glyph = "·"
            pill_style = "dim"

        # Compact summary like "2r 1✓ 1✗"
        parts: list[str] = []
        if counts.get("running"):
            parts.append(f"{counts['running']}▶")
        if counts.get("done"):
            parts.append(f"{counts['done']}✓")
        if counts.get("crashed"):
            parts.append(f"{counts['crashed']}✗")
        if counts.get("stalled"):
            parts.append(f"{counts['stalled']}⧗")
        if counts.get("stopped"):
            parts.append(f"{counts['stopped']}⊘")
        summary = " ".join(parts) if parts else f"{len(snaps)}"

        text = Text()
        text.append(f"{pill_glyph} ", style=pill_style)
        text.append(f"{title}  ", style="bold")
        text.append(f"[{summary}]", style="dim")
        return text

    def _collect_expanded_keys(self, tree: Tree) -> set[tuple]:
        """Walk the tree and remember which branch nodes are currently expanded."""
        expanded: set[tuple] = set()

        def walk(node: TreeNode) -> None:
            if node.data and isinstance(node.data, dict):
                kind = node.data.get("kind")
                key = node.data.get("key")
                if kind in ("type", "model") and key and node.is_expanded:
                    expanded.add(key)
            for child in node.children:
                walk(child)

        walk(tree.root)
        return expanded

    def _selected_run_name(self, tree: Tree) -> Optional[str]:
        cur = tree.cursor_node
        if cur is None or not cur.data:
            return None
        if cur.data.get("kind") == "leaf":
            return cur.data.get("name")
        return None

    def refresh_data(self) -> None:
        """Re-read all run dirs and rebuild the tree. Preserve expand / cursor / scroll."""
        self.snaps = [read_snapshot(d) for d in list_run_dirs(self.runs_dir)]

        # Surface manifest-declared runs that have never been launched (no dir yet)
        # as 'pending' placeholders, so they can be selected and `R`-restarted.
        existing_names = {s.name for s in self.snaps}
        try:
            import yaml as _yaml
            mdata = _yaml.safe_load(self.manifest.read_text()) or {}
            for exp in (mdata.get("experiments") or []):
                nm = exp.get("name")
                if not nm or nm in existing_names:
                    continue
                if exp.get("disabled"):
                    continue
                self.snaps.append(RunSnapshot(
                    name=nm,
                    group=exp.get("group", "ungrouped"),
                    exp_type=exp.get("type", "default"),
                    model=exp.get("model", "?"),
                    status="pending",
                    pid=None,
                    start_time=None,
                    total_queries=0,
                    done_queries=0,
                    elapsed_sec=0.0,
                    eta_sec=None,
                    run_dir=self.runs_dir / nm,
                    settings={},
                ))
        except Exception:
            pass

        self.snaps.sort(key=lambda s: (s.exp_type, s.model, s.name))

        tree = self.query_one("#run-tree", Tree)

        # Save state so rebuild doesn't yank the user's view.
        expanded_keys = self._collect_expanded_keys(tree)
        selected_name = self._selected_run_name(tree) or self._last_selected_run
        prev_scroll_y = tree.scroll_y

        tree.clear()
        self._leaf_nodes_by_name.clear()

        # Group: exp_type → model → [snaps]
        grouped: dict[str, dict[str, list[tuple[int, RunSnapshot]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for i, s in enumerate(self.snaps):
            grouped[s.exp_type][s.model].append((i, s))

        first = not self._first_refresh_done
        for exp_type in sorted(grouped.keys()):
            type_snaps = [s for m in grouped[exp_type].values() for _, s in m]
            type_key = ("type", exp_type)
            expand_type = (type_key in expanded_keys) or first
            type_node = tree.root.add(
                self._build_branch_label(exp_type, type_snaps),
                data={"kind": "type", "key": type_key},
                expand=expand_type,
            )
            for model in sorted(grouped[exp_type].keys()):
                runs = grouped[exp_type][model]
                model_snaps = [s for _, s in runs]
                model_key = ("model", exp_type, model)
                expand_model = model_key in expanded_keys
                model_node = type_node.add(
                    self._build_branch_label(model, model_snaps),
                    data={"kind": "model", "key": model_key},
                    expand=expand_model,
                )
                for idx, s in runs:
                    leaf = model_node.add_leaf(
                        self._build_leaf_label(s),
                        data={"kind": "leaf", "snap_idx": idx, "name": s.name},
                    )
                    self._leaf_nodes_by_name[s.name] = leaf

        self._first_refresh_done = True

        # Restore cursor to the same run if it still exists. Must be deferred
        # until after Textual processes the tree rebuild, otherwise the new
        # leaf's internal `_line` index is still unset and `select_node` would
        # end up pointing at nothing (cursor snaps back to the root).
        if selected_name and selected_name in self._leaf_nodes_by_name:
            target = self._leaf_nodes_by_name[selected_name]

            def _restore_cursor() -> None:
                tree.select_node(target)
                try:
                    tree.scroll_to(y=prev_scroll_y, animate=False)
                except Exception:
                    pass

            self.call_after_refresh(_restore_cursor)
        else:
            try:
                tree.scroll_to(y=prev_scroll_y, animate=False)
            except Exception:
                pass

        self.update_detail()

    # ---------------------- selection / detail ----------------------

    def selected_snapshot(self) -> Optional[RunSnapshot]:
        tree = self.query_one("#run-tree", Tree)
        cur = tree.cursor_node
        if cur is None or not cur.data:
            return None
        if cur.data.get("kind") != "leaf":
            return None
        idx = cur.data.get("snap_idx")
        if idx is None or idx < 0 or idx >= len(self.snaps):
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
            f"settings: [dim]{fmt_settings_badge(s.settings) or '-'}[/dim]\n"
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
            f"\n[dim]Tab switches focus between tree and log. j/k scroll the focused pane.[/dim]"
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
            self.query_one("#run-tree", Tree).focus()
            self.notify("focus: tree", severity="information")

    def on_log_scroll_focus_log(self, _: LogScroll.FocusLog) -> None:
        self.focus_on_log = True

    def on_tree_focus(self) -> None:
        self.focus_on_log = False

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        # Do NOT reset self._last_selected_run here. `update_detail` already
        # decides whether to follow-tail the log based on whether the selected
        # run name changed.
        self.update_detail()

    def action_context_down(self) -> None:
        if self.focus_on_log:
            self._log_scroll().scroll_relative(y=3, animate=False)
        else:
            self.query_one("#run-tree", Tree).action_cursor_down()

    def action_context_up(self) -> None:
        if self.focus_on_log:
            self._log_scroll().scroll_relative(y=-3, animate=False)
        else:
            self.query_one("#run-tree", Tree).action_cursor_up()

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
