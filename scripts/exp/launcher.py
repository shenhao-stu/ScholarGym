#!/usr/bin/env python3
"""ScholarGym experiment launcher.

Reads experiments.yaml, renders a per-run config.py, launches eval.py as a
detached background process (nohup-equivalent), and tracks state via state.json.

Subcommands:
    up      Start (or resume) experiments. Idempotent: existing runs auto-resume.
    down    Stop running experiments by sending SIGTERM (then SIGKILL after 5s).
    status  Print a one-shot status table for all known runs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Make `from exp.state import ...` work when invoked as a script
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from exp.state import (  # noqa: E402
    fmt_duration,
    list_run_dirs,
    read_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# YAML loading & config rendering
# ---------------------------------------------------------------------------

# Map yaml keys -> (config.py variable, value transformer)
YAML_TO_CONFIG = {
    "model": ("LLM_MODEL_NAME", lambda v: v),
    "is_local": ("IS_LOCAL_LLM", lambda v: bool(v)),
    "benchmark_jsonl": ("BENCHMARK_PATH", lambda v: v),
    "paper_db": ("PAPER_DB_PATH", lambda v: v),
    "bm25_path": ("BM25_PATH", lambda v: v),
    "workflow": ("EVAL_WORKFLOW", lambda v: v),
    "search_method": ("EVAL_SEARCH_METHOD", lambda v: v),
    "prompt_type": ("EVAL_PROMPT_TYPE", lambda v: v),
    "max_iterations": ("EVAL_MAX_ITERATIONS", lambda v: int(v)),
    "results_per_query": ("MAX_RESULTS_PER_QUERY", lambda v: int(v)),
    "browser_mode": ("BROWSER_MODE", lambda v: v),
    "enable_reasoning": ("ENABLE_REASONING", lambda v: bool(v)),
    "enable_structured_output": ("ENABLE_STRUCTURED_OUTPUT", lambda v: bool(v)),
    "planner_ablation": ("PLANNER_ABLATION", lambda v: bool(v)),
    "top_k": ("EVAL_TOP_K_VALUES", lambda v: list(v)),
}

CONFIG_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Required / commonly changed",
        [
            ("LLM_MODEL_NAME", "Change this to the model you want to evaluate."),
            ("IS_LOCAL_LLM", "Set True for local serving backends like Ollama or sglang."),
            ("BENCHMARK_PATH", "Benchmark file to evaluate."),
            ("EVAL_SEARCH_METHOD", "Retrieval method: 'bm25' or 'vector'."),
            ("EVAL_WORKFLOW", "Workflow: 'simple' or 'deep_research'."),
        ],
    ),
    (
        "Services",
        [
            ("QDRANT_URL", "Qdrant service endpoint."),
            ("OLLAMA_URL", "Local Ollama endpoint."),
        ],
    ),
    (
        "Data paths",
        [
            ("PAPER_DB_PATH", "Paper database JSON path."),
            ("EVAL_BASE_DIR", "Base output directory for legacy eval.py runs."),
        ],
    ),
    (
        "Retrieval / indexes",
        [
            ("BM25_PATH", "BM25 index path."),
            ("DEFAULT_SEARCH_METHOD", "Default retrieval backend outside explicit eval overrides."),
            ("QDRANT_COLLECTION_NAME", "Qdrant collection name."),
            ("QDRANT_EMBEDDING_MODEL", "Embedding model used for Qdrant indexing/querying."),
            ("VECTOR_SEARCH_TOP_K", "Vector retrieval fanout before later filtering."),
            ("BM25_SEARCH_TOP_K", "BM25 retrieval fanout before later filtering."),
        ],
    ),
    (
        "LLM — model + generation params",
        [
            ("DEVICE", "Compute device for local components."),
            ("LLM_GEN_PARAMS", "Generation parameters for the main evaluator model."),
            ("BROWSER_MAX_TOKENS", "Max tokens when browser-related LLM calls are made."),
            ("ENABLE_REASONING", "Usually keep default unless comparing reasoning ablations."),
            ("ENABLE_STRUCTURED_OUTPUT", "Use structured outputs when the provider supports them."),
            ("SAVE_AGENT_TRACES", "Persist detailed agent traces for debugging."),
            ("PLANNER_ABLATION", "Enable planner ablation mode for experiments."),
        ],
    ),
    (
        "Evaluation knobs",
        [
            ("EVAL_TOP_K_VALUES", "Top-k list used by the simple workflow."),
            ("EVAL_PROMPT_TYPE", "Prompt template family."),
            ("EVAL_MAX_ITERATIONS", "Iteration count for deep_research."),
            ("GT_RANK_CUTOFF", "Ground-truth rank cutoff for metrics."),
        ],
    ),
    (
        "Deep Research workflow",
        [
            ("CONTEXT_MAX_LENGTH_CHARS", "Context budget for long prompts."),
            ("ENABLE_LLM_FILTERING", "Whether to run selector-side LLM filtering."),
            ("LLM_FILTERING_BATCH_SIZE", "Batch size for selector filtering."),
            ("MAX_RESULTS_PER_QUERY", "Results retrieved per subquery in deep_research."),
            ("MAX_PAGES_PER_QUERY", "Browser page cap per subquery."),
            ("ENABLE_SUMMARIZATION", "Enable abstract/content summarization."),
            ("BROWSER_MODE", "Browser mode: 'NONE', 'PRE_ENRICH', 'REFRESH', or 'INCREMENTAL'."),
            ("SUMMARY_CACHE_PATH", "JSONL cache for summarizer outputs."),
            ("SUMMARY_ABSTRACT_CHAR_THRESHOLD", "Summarize abstracts longer than this threshold."),
        ],
    ),
    (
        "Summarization model",
        [
            ("SUMMARY_LLM_MODEL_NAME", "Model used by the summarizer."),
            ("SUMMARY_LLM_IS_LOCAL", "Whether summarizer model is served locally."),
            ("SUMMARY_LLM_GEN_PARAMS", "Generation parameters for the summarizer model."),
        ],
    ),
    (
        "Debug / tracing",
        [
            ("DEBUG", "Enable extra debug behaviors."),
            ("VERBOSE", "Print more intermediate workflow details."),
            ("CASE_STUDY_OUTPUT_DIR", "Directory for case-study artifacts."),
        ],
    ),
    (
        "Paper corpus",
        [
            ("TOTAL_PAPER_NUM", "Total number of papers in the corpus."),
        ],
    ),
]

CONFIG_FIELD_ORDER = [
    name
    for _, entries in CONFIG_SECTIONS
    for name, _ in entries
]


def load_manifest(path: Path) -> dict:
    with path.open() as f:
        manifest = yaml.safe_load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: top-level must be a mapping")
    if "experiments" not in manifest:
        raise ValueError(f"{path}: missing `experiments` key")

    # Validate uniqueness of names
    names = [e.get("name") for e in manifest["experiments"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate experiment names in manifest")
    if any(not n for n in names):
        raise ValueError("Every experiment must have a `name`")

    return manifest


def merged_settings(defaults: dict, exp: dict) -> dict:
    """Merge defaults + experiment override (shallow merge, exp wins)."""
    merged = dict(defaults or {})
    merged.update({k: v for k, v in exp.items() if k not in ("name", "group", "env")})
    return merged


def _load_base_config_values() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "code" / "config.py"
    spec = importlib.util.spec_from_file_location("scholargym_base_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    values: dict[str, Any] = {}
    missing_fields: list[str] = []
    for name in CONFIG_FIELD_ORDER:
        if not hasattr(module, name):
            missing_fields.append(name)
            continue
        values[name] = getattr(module, name)

    if missing_fields:
        missing = ", ".join(missing_fields)
        raise RuntimeError(f"code/config.py is missing expected config fields: {missing}")

    return values


def _render_config_lines(values: dict[str, Any], *, header_lines: list[str]) -> list[str]:
    lines = list(header_lines)

    for section_name, entries in CONFIG_SECTIONS:
        lines.extend([
            "# =============================================================================",
            f"# {section_name}",
            "# =============================================================================",
            "",
        ])
        for name, comment in entries:
            if comment:
                lines.append(f"# {comment}")
            lines.append(f"{name} = {repr(values[name])}")
            lines.append("")

    return lines


def render_config(merged: dict, target: Path) -> None:
    """Write a fully resolved config.py snapshot for this run."""
    resolved = _load_base_config_values()
    for key, val in merged.items():
        if key not in YAML_TO_CONFIG:
            continue
        config_var, transform = YAML_TO_CONFIG[key]
        resolved[config_var] = transform(val)

    lines = _render_config_lines(
        resolved,
        header_lines=[
            "# Auto-generated by scripts/exp/launcher.py — do not edit by hand.",
            "# Resolved config snapshot for this run.",
            "# This file uses the same layout as code/config.py for easier inspection.",
            "",
        ],
    )
    target.write_text("\n".join(lines))


def expand_env(value: Any) -> Any:
    """Resolve ${VAR} references against os.environ. Recurses into dicts."""
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            return os.environ.get(m.group(1), m.group(0))
        return re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}", repl, value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


def count_benchmark_lines(benchmark_path: Path) -> int:
    if not benchmark_path.exists():
        return 0
    with benchmark_path.open("rb") as f:
        return sum(1 for _ in f)


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def _is_pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def launch_one(
    exp: dict,
    defaults: dict,
    runs_root: Path,
    fresh: bool,
) -> None:
    name = exp["name"]
    group = exp.get("group", "ungrouped")
    exp_type = exp.get("type", "default")
    run_dir = runs_root / name

    # --fresh: nuke existing run dir
    if fresh and run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)
        print(f"[fresh] removed existing {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)

    # Merge defaults + per-experiment overrides
    merged = merged_settings(defaults, exp)

    config_path = run_dir / "config.py"
    cp_path = run_dir / "detailed_results.jsonl"
    has_checkpoint = cp_path.exists() and cp_path.stat().st_size > 0

    # Resume safety: if checkpoint exists and config.py is also there, keep config.py
    # (avoids polluting an in-progress experiment with newer code defaults)
    if has_checkpoint and config_path.exists():
        print(f"[resume] {name}: keeping existing config.py and resuming from checkpoint ({_count_lines(cp_path)} lines)")
    else:
        render_config(merged, config_path)
        print(f"[render] {name}: wrote {config_path}")

    # Always update manifest snapshot for transparency
    (run_dir / "manifest.yaml").write_text(yaml.safe_dump(exp, sort_keys=False))

    # Refuse to start a duplicate process
    state_file = run_dir / "state.json"
    if state_file.exists():
        try:
            existing = json.loads(state_file.read_text())
            if _is_pid_alive(existing.get("pid")):
                print(f"[skip ] {name}: already running (pid={existing['pid']})")
                return
        except (json.JSONDecodeError, OSError):
            pass

    # Compute total queries from the benchmark referenced in the merged config
    bench_path = PROJECT_ROOT / merged.get("benchmark_jsonl", "data/scholargym_bench.jsonl")
    total_queries = count_benchmark_lines(bench_path)

    # Build command
    cmd = [
        sys.executable,
        "code/eval.py",
        "--config", str(config_path.relative_to(PROJECT_ROOT)),
        "--run_dir", str(run_dir.relative_to(PROJECT_ROOT)),
    ]

    # Environment
    proc_env = os.environ.copy()
    proc_env.setdefault("no_proxy", "localhost,127.0.0.1")
    yaml_env = expand_env(exp.get("env") or {})
    proc_env.update(yaml_env)

    # Launch detached, redirect stdout+stderr to run.log
    log_path = run_dir / "run.log"
    log_fp = log_path.open("ab")
    log_fp.write(f"\n===== launched at {datetime.now().isoformat(timespec='seconds')} =====\n".encode())
    log_fp.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        env=proc_env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )

    state = {
        "name": name,
        "group": group,
        "type": exp_type,
        "pid": proc.pid,
        "command": " ".join(cmd),
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "total_queries": total_queries,
        "status": "running",
        "exit_code": None,
        "stopped_at": None,
        "log_file": str(log_path.relative_to(PROJECT_ROOT)),
    }
    state_file.write_text(json.dumps(state, indent=2))
    print(f"[start] {name}: pid={proc.pid}, total={total_queries}, log={log_path.relative_to(PROJECT_ROOT)}")


def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def stop_one(run_dir: Path) -> None:
    state_file = run_dir / "state.json"
    if not state_file.exists():
        print(f"[skip ] {run_dir.name}: no state.json")
        return
    state = json.loads(state_file.read_text())
    pid = state.get("pid")
    if not pid or not _is_pid_alive(pid):
        print(f"[skip ] {run_dir.name}: not running")
        state["status"] = state.get("status") if state.get("status") in ("done", "crashed") else "stopped"
        state_file.write_text(json.dumps(state, indent=2))
        return

    print(f"[stop ] {run_dir.name}: SIGTERM pid={pid}")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    # Wait up to 5s
    for _ in range(50):
        if not _is_pid_alive(pid):
            break
        time.sleep(0.1)
    else:
        print(f"[kill ] {run_dir.name}: SIGKILL pid={pid}")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    state["status"] = "stopped"
    state["stopped_at"] = datetime.now().isoformat(timespec="seconds")
    state_file.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_up(args: argparse.Namespace) -> None:
    manifest = load_manifest(Path(args.manifest))
    runs_root = PROJECT_ROOT / manifest.get("runs_dir", "runs")
    runs_root.mkdir(exist_ok=True)
    defaults = manifest.get("defaults", {})

    only = set(args.only.split(",")) if args.only else None
    for exp in manifest["experiments"]:
        if only and exp["name"] not in only:
            continue
        launch_one(exp, defaults, runs_root, fresh=args.fresh)


def cmd_down(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        runs_root = PROJECT_ROOT / manifest.get("runs_dir", "runs")
    else:
        runs_root = PROJECT_ROOT / "runs"

    only = set(args.only.split(",")) if args.only else None
    for run_dir in list_run_dirs(runs_root):
        if only and run_dir.name not in only:
            continue
        stop_one(run_dir)


def cmd_restart(args: argparse.Namespace) -> None:
    """down + up for the given names. Equivalent to running both in sequence."""
    manifest = load_manifest(Path(args.manifest))
    runs_root = PROJECT_ROOT / manifest.get("runs_dir", "runs")
    runs_root.mkdir(exist_ok=True)
    defaults = manifest.get("defaults", {})

    only = set(args.only.split(",")) if args.only else None

    # Stop matching runs that are still alive
    for run_dir in list_run_dirs(runs_root):
        if only and run_dir.name not in only:
            continue
        stop_one(run_dir)

    # Then launch them again
    for exp in manifest["experiments"]:
        if only and exp["name"] not in only:
            continue
        launch_one(exp, defaults, runs_root, fresh=args.fresh)


def cmd_status(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        runs_root = PROJECT_ROOT / manifest.get("runs_dir", "runs")
    else:
        runs_root = PROJECT_ROOT / "runs"

    snaps = [read_snapshot(d) for d in list_run_dirs(runs_root)]
    if not snaps:
        print(f"No runs found under {runs_root}")
        return

    # Group printing: type -> group -> run
    snaps.sort(key=lambda s: (s.exp_type, s.group, s.name))
    print(f"{'NAME':24} {'TYPE':10} {'STATUS':10} {'PROGRESS':14} {'ELAPSED':>8} {'ETA':>8}  METRIC")
    print("-" * 96)
    last_type = None
    last_group = None
    for s in snaps:
        if s.exp_type != last_type:
            print(f"## {s.exp_type} ##")
            last_type = s.exp_type
            last_group = None
        if s.group != last_group:
            print(f"== {s.group} ==")
            last_group = s.group
        prog = f"{s.done_queries}/{s.total_queries} ({s.progress_ratio*100:.1f}%)"
        metric = ""
        if s.last_metric:
            m = s.last_metric
            metric = f"R={m['sel_r']:.2f} P={m['sel_p']:.2f}"
        if s.anomaly:
            metric = f"⚠ {s.anomaly} {metric}"
        print(f"{s.name:24} {s.exp_type:10} {s.status:10} {prog:14} {fmt_duration(s.elapsed_sec):>8} {fmt_duration(s.eta_sec):>8}  {metric}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="experiments.yaml", help="Path to experiments.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_up = sub.add_parser("up", help="Launch (or resume) experiments")
    p_up.add_argument("--only", default=None, help="Comma-separated names to launch (default: all)")
    p_up.add_argument("--fresh", action="store_true", help="Wipe existing run dirs before launch")
    p_up.set_defaults(func=cmd_up)

    p_down = sub.add_parser("down", help="Stop running experiments")
    p_down.add_argument("--only", default=None, help="Comma-separated names to stop (default: all)")
    p_down.set_defaults(func=cmd_down)

    p_restart = sub.add_parser("restart", help="Stop then re-launch experiments")
    p_restart.add_argument("--only", default=None, help="Comma-separated names to restart (default: all)")
    p_restart.add_argument("--fresh", action="store_true", help="Wipe run dirs before re-launching")
    p_restart.set_defaults(func=cmd_restart)

    p_status = sub.add_parser("status", help="Print one-shot status table")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
