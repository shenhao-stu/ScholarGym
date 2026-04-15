#!/usr/bin/env python3
"""One-shot migration: copy existing eval_results_rerun_*/ checkpoints into runs/.

The new layout uses runs/<name>/ with state.json, detailed_results.jsonl, etc.
This script does a non-destructive copy so the old directories remain until the
user verifies the migration manually.

Usage:
    python scripts/exp/migrate.py [--runs-dir runs] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

# Hard-coded mapping of old dirs → new run names. These match what's in
# experiments.yaml. If the user later renames experiments, they should
# rerun migrate or move dirs by hand.
MIGRATIONS = [
    {
        "new_name": "qwen3-8b-none",
        "group": "qwen3-8b",
        "type": "main",
        "old_dir": "eval_results_rerun_qwen3_8b_mainexp/qwen3-8b_complex_bm25_deep_research_maxq-10_iter-5_reasoning_non-structured_NONE",
    },
    {
        "new_name": "qwen3-8b-refresh",
        "group": "qwen3-8b",
        "type": "main",
        "old_dir": "eval_results_rerun_qwen3_8b_mainexp/qwen3-8b_complex_bm25_deep_research_maxq-10_iter-5_reasoning_non-structured_REFRESH",
    },
    {
        "new_name": "glm51-none",
        "group": "glm-5.1",
        "type": "main",
        "old_dir": "eval_results_rerun_glm51_mainexp/glm-5.1_complex_bm25_deep_research_maxq-10_iter-5_reasoning_non-structured_NONE",
    },
]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def migrate_one(project_root: Path, runs_root: Path, entry: dict, dry_run: bool) -> None:
    name = entry["new_name"]
    old = project_root / entry["old_dir"]
    new = runs_root / name

    if not old.exists():
        print(f"[skip] {name}: source not found ({old})")
        return

    if new.exists() and (new / "detailed_results.jsonl").exists():
        print(f"[skip] {name}: target already exists with checkpoint, not overwriting")
        return

    cp = old / "detailed_results.jsonl"
    n_lines = count_lines(cp)
    mtime = datetime.fromtimestamp(old.stat().st_mtime).isoformat(timespec="seconds")

    print(
        f"[migrate] {name}  ←  {entry['old_dir']}\n"
        f"          checkpoint lines = {n_lines}"
    )
    if dry_run:
        return

    new.mkdir(parents=True, exist_ok=True)

    if cp.exists():
        shutil.copy2(cp, new / "detailed_results.jsonl")
    # NOTE: we deliberately do NOT copy the old config.py. The launcher will
    # render a fresh one from experiments.yaml on the next `up`, ensuring
    # the manifest is the single source of truth. (Old config.py was the full
    # base config, not a CLI-aware snapshot, so it could mismatch the checkpoint.)

    state = {
        "name": name,
        "group": entry["group"],
        "type": entry.get("type", "main"),
        "pid": None,
        "command": None,
        "start_time": mtime,
        "total_queries": 200,  # default; launcher recomputes from benchmark on next up
        "status": "stopped",
        "exit_code": None,
        "stopped_at": datetime.now().isoformat(timespec="seconds"),
        "migrated_from": str(entry["old_dir"]),
    }
    (new / "state.json").write_text(json.dumps(state, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs", help="Target runs/ directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    runs_root = project_root / args.runs_dir

    if not args.dry_run:
        runs_root.mkdir(exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Runs root:    {runs_root}")
    print()

    for entry in MIGRATIONS:
        migrate_one(project_root, runs_root, entry, args.dry_run)

    print("\nDone." + (" (dry-run, nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
