"""Pure helpers for reading/deriving experiment run state.

A "run" is a directory under runs/ created by the launcher. This module
knows how to read state.json, count checkpoint progress, derive ETA, and
detect anomalies by tailing the log file. Everything here is read-only.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional

# Patterns we treat as fatal anomalies when seen in run.log
ANOMALY_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\):"),
    re.compile(r"\bAPIError\b"),
    re.compile(r"\bConnectionError\b"),
    re.compile(r"\bRateLimitError\b"),
    re.compile(r"\bTimeoutError\b"),
    re.compile(r"\bRuntimeError\b"),
    re.compile(r"openai\.[A-Za-z]*Error"),
]

# Pattern to extract the last "Cumulative Metrics after Iter N" line
METRIC_LINE_RE = re.compile(
    r"Cumulative Metrics after Iter (\d+)\].*"
    r"Retrieval Recall: ([\d.]+).*Retrieval Precision: ([\d.]+).*"
    r"Selection Recall: ([\d.]+).*Selection Precision: ([\d.]+)"
)


@dataclass
class RunSnapshot:
    """A point-in-time view of one experiment's progress."""

    name: str
    group: str
    exp_type: str
    status: str  # running | done | crashed | stopped | stalled | unknown
    pid: Optional[int]
    start_time: Optional[datetime]
    total_queries: int
    done_queries: int
    elapsed_sec: float
    eta_sec: Optional[float]
    last_metric: Optional[dict] = None  # {iter, ret_r, ret_p, sel_r, sel_p}
    anomaly: Optional[str] = None  # short label e.g. "APIError"
    last_progress_at: Optional[datetime] = None
    run_dir: Path = field(default_factory=Path)

    @property
    def progress_ratio(self) -> float:
        if self.total_queries <= 0:
            return 0.0
        return min(1.0, self.done_queries / self.total_queries)


def _is_pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _load_state(run_dir: Path) -> dict:
    state_file = run_dir / "state.json"
    if not state_file.exists():
        return {}
    try:
        return json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _load_manifest(run_dir: Path) -> dict:
    manifest_file = run_dir / "manifest.yaml"
    if not manifest_file.exists():
        return {}
    try:
        data = yaml.safe_load(manifest_file.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _count_checkpoint(run_dir: Path) -> int:
    cp = run_dir / "detailed_results.jsonl"
    if not cp.exists():
        return 0
    try:
        with cp.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _checkpoint_mtime(run_dir: Path) -> Optional[datetime]:
    cp = run_dir / "detailed_results.jsonl"
    if not cp.exists():
        return None
    try:
        return datetime.fromtimestamp(cp.stat().st_mtime)
    except OSError:
        return None


def _tail_log(run_dir: Path, max_bytes: int = 64 * 1024) -> str:
    log = run_dir / "run.log"
    if not log.exists():
        return ""
    try:
        size = log.stat().st_size
        with log.open("rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, 2)
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _detect_anomaly(log_tail: str) -> Optional[str]:
    if not log_tail:
        return None
    for pattern in ANOMALY_PATTERNS:
        m = pattern.search(log_tail)
        if m:
            return m.group(0)
    return None


def _extract_last_metric(log_tail: str) -> Optional[dict]:
    if not log_tail:
        return None
    matches = list(METRIC_LINE_RE.finditer(log_tail))
    if not matches:
        return None
    m = matches[-1]
    return {
        "iter": int(m.group(1)),
        "ret_r": float(m.group(2)),
        "ret_p": float(m.group(3)),
        "sel_r": float(m.group(4)),
        "sel_p": float(m.group(5)),
    }


def extract_metric_history(log_tail: str) -> list[dict]:
    """Return all `Cumulative Metrics after Iter N` entries seen in the tail.

    Used by the TUI / web UI to draw sparklines of metric trajectories.
    Note: this only sees what's in the in-memory tail (last 64KB by default),
    so on long-running experiments it shows recent history, not full history.
    """
    if not log_tail:
        return []
    out: list[dict] = []
    for m in METRIC_LINE_RE.finditer(log_tail):
        out.append(
            {
                "iter": int(m.group(1)),
                "ret_r": float(m.group(2)),
                "ret_p": float(m.group(3)),
                "sel_r": float(m.group(4)),
                "sel_p": float(m.group(5)),
            }
        )
    return out


def read_log_tail_lines(run_dir: Path, n_lines: int = 40, max_bytes: int = 256 * 1024) -> list[str]:
    """Return the last `n_lines` lines from run.log."""
    log = run_dir / "run.log"
    if not log.exists():
        return []
    try:
        size = log.stat().st_size
        with log.open("rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, 2)
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = data.splitlines()
    return lines[-n_lines:]


def sparkline(values: list[float], width: int = 20, vmin: float = 0.0, vmax: float = 1.0) -> str:
    """Render a unicode sparkline from a sequence of values.

    Values outside [vmin, vmax] are clamped. The sequence is downsampled to
    `width` points by taking evenly spaced samples (or padded with spaces).
    """
    blocks = "▁▂▃▄▅▆▇█"
    if not values:
        return " " * width
    if len(values) > width:
        # Downsample by taking every k-th value
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = list(values)
    out = []
    span = max(vmax - vmin, 1e-9)
    for v in sampled:
        v = max(vmin, min(vmax, v))
        idx = int((v - vmin) / span * (len(blocks) - 1))
        out.append(blocks[idx])
    # Left-pad with spaces if shorter than width
    if len(out) < width:
        out = [" "] * (width - len(out)) + out
    return "".join(out)


def read_snapshot(run_dir: Path) -> RunSnapshot:
    """Build a RunSnapshot for a single run directory."""
    run_dir = Path(run_dir)
    state = _load_state(run_dir)
    manifest = _load_manifest(run_dir)

    name = state.get("name", manifest.get("name", run_dir.name))
    group = state.get("group", manifest.get("group", "ungrouped"))
    exp_type = state.get("type", state.get("exp_type", manifest.get("type", "default")))
    pid = state.get("pid")
    total = int(state.get("total_queries", 0) or 0)
    declared_status = state.get("status", "unknown")

    done = _count_checkpoint(run_dir)
    last_progress_at = _checkpoint_mtime(run_dir)

    start_str = state.get("start_time")
    start_time: Optional[datetime] = None
    if start_str:
        try:
            start_time = datetime.fromisoformat(start_str)
        except ValueError:
            start_time = None

    elapsed = (datetime.now() - start_time).total_seconds() if start_time else 0.0
    eta: Optional[float] = None
    if done > 0 and total > done and elapsed > 0:
        per_query = elapsed / done
        eta = per_query * (total - done)

    log_tail = _tail_log(run_dir)
    anomaly = _detect_anomaly(log_tail)
    last_metric = _extract_last_metric(log_tail)

    # Status resolution
    results_done = (run_dir / "results.json").exists()
    pid_alive = _is_pid_alive(pid)

    if declared_status == "stopped" and not pid_alive:
        status = "stopped"
    elif results_done:
        status = "done"
    elif pid_alive:
        # Stall detection: pid alive but no progress for >15 minutes since launch.
        # We measure from max(start_time, last_progress_at) so a freshly launched
        # resume is not flagged stalled because its checkpoint mtime is old.
        ref_times = [t for t in (start_time, last_progress_at) if t is not None]
        ref = max(ref_times) if ref_times else None
        if ref and (datetime.now() - ref).total_seconds() > 900:
            status = "stalled"
        else:
            status = "running"
    elif declared_status == "running" and not pid_alive:
        status = "crashed"
    else:
        status = declared_status or "unknown"

    return RunSnapshot(
        name=name,
        group=group,
        exp_type=exp_type,
        status=status,
        pid=pid,
        start_time=start_time,
        total_queries=total,
        done_queries=done,
        elapsed_sec=elapsed,
        eta_sec=eta,
        last_metric=last_metric,
        anomaly=anomaly,
        last_progress_at=last_progress_at,
        run_dir=run_dir,
    )


def list_run_dirs(runs_root: Path) -> list[Path]:
    """Return all subdirectories under runs/ that look like a run dir."""
    runs_root = Path(runs_root)
    if not runs_root.exists():
        return []
    return sorted(
        [d for d in runs_root.iterdir() if d.is_dir() and (d / "state.json").exists()],
        key=lambda p: p.name,
    )


def fmt_duration(sec: Optional[float]) -> str:
    if sec is None or sec < 0:
        return "--"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"
