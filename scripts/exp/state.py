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
    model: str  # concrete model id, e.g. 'qwen3-8b' / 'claude-sonnet-4-6'
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
    settings: dict = field(default_factory=dict)  # from manifest.yaml `settings` block

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
        st = state_file.stat()
    except OSError:
        return {}
    key = str(state_file)
    cached = _STATE_CACHE.get(key)
    if cached is not None and cached[0] == st.st_mtime and cached[1] == st.st_size:
        return cached[2]
    try:
        data = json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    _STATE_CACHE[key] = (st.st_mtime, st.st_size, data)
    return data


def _load_manifest(run_dir: Path) -> dict:
    manifest_file = run_dir / "manifest.yaml"
    if not manifest_file.exists():
        return {}
    try:
        st = manifest_file.stat()
    except OSError:
        return {}
    key = str(manifest_file)
    cached = _MANIFEST_CACHE.get(key)
    if cached is not None and cached[0] == st.st_mtime and cached[1] == st.st_size:
        return cached[2]
    try:
        data = yaml.safe_load(manifest_file.read_text())
        data = data if isinstance(data, dict) else {}
    except Exception:
        return {}
    _MANIFEST_CACHE[key] = (st.st_mtime, st.st_size, data)
    return data


# Per-file caches keyed by path → (mtime, size, parsed). mtime+size invalidation
# avoids re-parsing state.json / manifest.yaml on every 5s refresh tick.
_STATE_CACHE: dict[str, tuple[float, int, dict]] = {}
_MANIFEST_CACHE: dict[str, tuple[float, int, dict]] = {}


def _load_config_fallback_settings(run_dir: Path) -> dict:
    """Fallback: derive the 7-dim settings from config.py if manifest lacks a `settings` block.

    Used for legacy runs launched before the manifest schema was introduced.
    """
    config_file = run_dir / "config.py"
    if not config_file.exists():
        return {}
    text = config_file.read_text(errors="replace")

    def _grep(var: str) -> Optional[str]:
        m = re.search(rf"^\s*{re.escape(var)}\s*=\s*(.+?)\s*$", text, re.MULTILINE)
        return m.group(1) if m else None

    def _as_bool(expr: Optional[str]) -> bool:
        return bool(expr) and expr.strip().lower() in {"true", "1", "'true'", '"true"'}

    def _as_int(expr: Optional[str], default: int) -> int:
        if not expr:
            return default
        try:
            return int(expr)
        except ValueError:
            return default

    def _strip_quotes(expr: Optional[str], default: str) -> str:
        if not expr:
            return default
        return expr.strip().strip("'\"")

    bench = _strip_quotes(_grep("BENCHMARK_PATH"), "data/test_fast.jsonl")
    dataset = "hard" if "test_hard" in bench else ("fast" if "test_fast" in bench else bench)
    workflow_raw = _strip_quotes(_grep("EVAL_WORKFLOW"), "deep_research")
    return {
        "thinking": _as_bool(_grep("ENABLE_REASONING")),
        "browser": _strip_quotes(_grep("BROWSER_MODE"), "NONE").upper(),
        "dataset": dataset,
        "search": _strip_quotes(_grep("EVAL_SEARCH_METHOD"), "bm25"),
        "iterations": _as_int(_grep("EVAL_MAX_ITERATIONS"), 5),
        "memory": not _as_bool(_grep("PLANNER_ABLATION")),
        "workflow": "direct" if workflow_raw == "simple" else "deep_research",
    }


def _count_checkpoint(run_dir: Path) -> int:
    cp = run_dir / "detailed_results.jsonl"
    if not cp.exists():
        return 0
    try:
        st = cp.stat()
    except OSError:
        return 0
    key = str(cp)
    cached = _CHECKPOINT_CACHE.get(key)
    if cached is not None and cached[0] == st.st_mtime and cached[1] == st.st_size:
        return cached[2]
    # File changed (or first read): do a fast buffered newline count.
    try:
        count = 0
        with cp.open("rb") as f:
            while True:
                chunk = f.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                count += chunk.count(b"\n")
    except OSError:
        return 0
    _CHECKPOINT_CACHE[key] = (st.st_mtime, st.st_size, count)
    return count


# Cache to avoid re-counting `detailed_results.jsonl` every refresh tick
# when the file hasn't changed. Keyed by path; value is (mtime, size, count).
_CHECKPOINT_CACHE: dict[str, tuple[float, int, int]] = {}


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
        st = log.stat()
    except OSError:
        return ""
    key = (str(log), max_bytes)
    cached = _LOG_TAIL_CACHE.get(key)
    if cached is not None and cached[0] == st.st_mtime and cached[1] == st.st_size:
        return cached[2]
    try:
        with log.open("rb") as f:
            if st.st_size > max_bytes:
                f.seek(-max_bytes, 2)
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    _LOG_TAIL_CACHE[key] = (st.st_mtime, st.st_size, data)
    return data


# Cache for `_tail_log` — reduces repeated 64KB-per-run reads every refresh
# when the log hasn't changed. Bounded implicitly by one entry per run.
_LOG_TAIL_CACHE: dict[tuple[str, int], tuple[float, int, str]] = {}


def _latest_launch_tail(log_tail: str) -> str:
    if not log_tail:
        return ""
    marker = "===== launched at "
    idx = log_tail.rfind(marker)
    if idx == -1:
        return log_tail
    return log_tail[idx:]


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
    model = state.get("model") or manifest.get("model") or group
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
    current_launch_tail = _latest_launch_tail(log_tail)
    anomaly = _detect_anomaly(current_launch_tail)
    last_metric = _extract_last_metric(current_launch_tail)

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
        # Progress ≥95% but results.json absent → treat as effectively done
        # (likely crashed only during final finalize step). Everything below
        # that threshold is a real crash.
        if total > 0 and done / total >= 0.95:
            status = "done"
        else:
            status = "crashed"
    else:
        status = declared_status or "unknown"

    return RunSnapshot(
        name=name,
        group=group,
        exp_type=exp_type,
        model=model,
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
        settings=manifest.get("settings") or _load_config_fallback_settings(run_dir),
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


def fmt_settings_badge(s: dict) -> str:
    """Render the 7-dim settings block as a compact ordered badge string.

    Format: `{browser} [{dataset}] [think] [{search}] [iter{N}] [nomem] [direct]`
    Default values are elided: dataset=fast, search=bm25, iterations=5,
    memory=true, workflow=deep_research. `vector` is displayed as `dense`.
    """
    if not s:
        return ""
    parts = [str(s.get("browser", "?"))[:7].lower()]
    dataset = s.get("dataset", "fast")
    if dataset != "fast":
        parts.append(dataset)
    if s.get("thinking"):
        parts.append("think")
    search = s.get("search", "bm25")
    if search != "bm25":
        search_display = {"vector": "dense"}.get(search, search)
        parts.append(search_display)
    it = s.get("iterations", 5)
    if it != 5:
        parts.append(f"iter{it}")
    if s.get("memory") is False:
        parts.append("nomem")
    workflow = s.get("workflow", "deep_research")
    if workflow == "direct":
        parts.append("direct")
    return " ".join(parts)
