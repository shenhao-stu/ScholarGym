"""Estimate per-run / aggregate GPU runtime for ScholarGym experiments.

Scans run directories under `runs/` and `runs_iter25/` (including
`_archived/*` for post-merge shards) and reports three time numbers per run:

  * active_sec   — sum of `total_during` across all (query, iter) tuples in
                   `detailed_results.jsonl`. This is the actual LLM compute
                   time, immune to retries / sleeps / network stalls.
  * log_wall_sec — wall clock derived from `run.log` by summing each
                   `===== launched at <ISO>` segment up to the latest
                   timestamp seen before the next launch marker (or EOF).
                   Excludes time the process was not running across reruns.
  * state_wall_sec — coarse fallback: state.json `start_time` to either
                   results.json mtime, last checkpoint mtime, or now. Less
                   accurate than log_wall_sec but works when run.log was
                   rotated or merged away.

Merged shard runs (state.json has `merged_from`) have no run.log; their
active_sec still works because detailed_results.jsonl is the merged file.
The archived shards under `_archived/*` are scanned too so wall time is
not lost.

Usage:
    python scripts/exp/runtime_summary.py
    python scripts/exp/runtime_summary.py --runs-root runs_iter25 \
        --group-by model --csv /tmp/scholargym_runtime.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = ["runs", "runs_iter25"]
DEFAULT_MANIFESTS = ["experiments.yaml", "experiments_iter25.yaml"]

LAUNCH_RE = re.compile(r"^===== launched at (\S+) =====")
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})")
SHARD_SUFFIX_RE = re.compile(r"-?shard\d+of\d+$")


def _experiment_slug(name: str) -> str:
    """Strip a `-shardKofN` suffix so all shards of one experiment share a slug."""
    return SHARD_SUFFIX_RE.sub("", name).rstrip("-_")


@dataclass
class RunInfo:
    name: str
    group: str
    exp_type: str
    model: str
    settings: dict = field(default_factory=dict)
    total_queries: int = 0
    done_queries: int = 0
    active_sec: float = 0.0
    log_wall_sec: float = 0.0
    state_wall_sec: float = 0.0
    n_launches: int = 0
    is_merged: bool = False
    is_archived_shard: bool = False
    has_log: bool = False
    has_detailed: bool = False
    run_dir: Path = field(default_factory=Path)


def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = s.replace(" ", "T")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _read_state(run_dir: Path) -> dict:
    f = run_dir / "state.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _read_manifest_settings(run_dir: Path) -> dict:
    f = run_dir / "manifest.yaml"
    if not f.exists():
        return {}
    try:
        import yaml
        data = yaml.safe_load(f.read_text()) or {}
        return data.get("settings") or {}
    except Exception:
        return {}


def _sum_total_during(detailed_jsonl: Path) -> tuple[float, int]:
    """Return (sum of per-iter total_during across all queries, num queries)."""
    if not detailed_jsonl.exists():
        return 0.0, 0
    total = 0.0
    n = 0
    try:
        with detailed_jsonl.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                for it in rec.get("iteration_results") or []:
                    v = it.get("total_during")
                    if isinstance(v, (int, float)) and v > 0:
                        total += float(v)
    except OSError:
        return 0.0, n
    return total, n


def _log_wall_seconds(run_log: Path) -> tuple[float, int]:
    """Sum per-launch active intervals from run.log.

    A launch interval starts at `===== launched at <ISO>` and ends at the
    last `YYYY-MM-DD HH:MM:SS` timestamp before the next launch marker (or
    EOF). Lines without a parseable timestamp are ignored.
    """
    if not run_log.exists():
        return 0.0, 0
    try:
        text = run_log.read_text(errors="replace")
    except OSError:
        return 0.0, 0

    segments: list[tuple[datetime, list[datetime]]] = []
    cur_start: Optional[datetime] = None
    cur_ends: list[datetime] = []
    for line in text.splitlines():
        m = LAUNCH_RE.match(line)
        if m:
            if cur_start is not None:
                segments.append((cur_start, cur_ends))
            cur_start = _parse_iso(m.group(1))
            cur_ends = []
            continue
        if cur_start is None:
            continue
        tm = TS_RE.search(line)
        if tm:
            ts = _parse_iso(tm.group(1))
            if ts is not None:
                cur_ends.append(ts)
    if cur_start is not None:
        segments.append((cur_start, cur_ends))

    total = 0.0
    n_launches = 0
    for start, ends in segments:
        if not ends:
            continue
        end = max(ends)
        if end <= start:
            continue
        total += (end - start).total_seconds()
        n_launches += 1
    return total, n_launches


def _state_wall_seconds(run_dir: Path, state: dict) -> float:
    start = _parse_iso(state.get("start_time") or "")
    if start is None:
        return 0.0
    end_candidates = []
    for fname in ("results.json", "detailed_results.jsonl", "merge_summary.json"):
        f = run_dir / fname
        if f.exists():
            try:
                end_candidates.append(datetime.fromtimestamp(f.stat().st_mtime))
            except OSError:
                pass
    stopped = state.get("stopped_at")
    if stopped:
        ts = _parse_iso(stopped)
        if ts is not None:
            end_candidates.append(ts)
    if state.get("status") == "running":
        end_candidates.append(datetime.now())
    if not end_candidates:
        return 0.0
    end = max(end_candidates)
    return max(0.0, (end - start).total_seconds())


_ARCHIVE_DUP_SUFFIXES = ("_postshardprep_", "_preshard_", "_pre_shard_")


def _iter_run_dirs(roots: Iterable[Path], include_archived_dups: bool = False) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            if d.name == "_archived":
                for sub in sorted(d.iterdir()):
                    if not (sub.is_dir() and (sub / "state.json").exists()):
                        continue
                    if not include_archived_dups and any(
                        s in sub.name for s in _ARCHIVE_DUP_SUFFIXES
                    ):
                        # Snapshots taken during shard-prep migration; duplicates
                        # of the live or _postmerge_ shard dirs.
                        continue
                    out.append(sub)
                continue
            if (d / "state.json").exists():
                out.append(d)
    return out


def collect(roots: list[Path], include_archived_dups: bool = False) -> list[RunInfo]:
    infos: list[RunInfo] = []
    for run_dir in _iter_run_dirs(roots, include_archived_dups):
        state = _read_state(run_dir)
        if not state and not (run_dir / "manifest.yaml").exists():
            continue

        settings = _read_manifest_settings(run_dir)
        info = RunInfo(
            name=state.get("name", run_dir.name),
            group=state.get("group", "ungrouped"),
            exp_type=state.get("type", state.get("exp_type", "default")),
            model=state.get("model") or state.get("group") or "?",
            settings=settings,
            total_queries=int(state.get("total_queries") or 0),
            run_dir=run_dir,
        )
        info.is_merged = bool(state.get("merged_from"))
        info.is_archived_shard = "_archived" in run_dir.parts
        info.has_log = (run_dir / "run.log").exists()
        info.has_detailed = (run_dir / "detailed_results.jsonl").exists()

        info.active_sec, info.done_queries = _sum_total_during(
            run_dir / "detailed_results.jsonl"
        )
        info.log_wall_sec, info.n_launches = _log_wall_seconds(run_dir / "run.log")
        info.state_wall_sec = _state_wall_seconds(run_dir, state)
        infos.append(info)
    return infos


def fmt_dur(sec: float) -> str:
    if sec is None or sec <= 0:
        return "--"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _flag_str(info: RunInfo) -> str:
    flags = []
    if info.is_archived_shard:
        flags.append("arch")
    if info.is_merged:
        flags.append("merged")
    if not info.has_log:
        flags.append("no-log")
    if not info.has_detailed:
        flags.append("no-detail")
    return ",".join(flags) or "-"


def print_per_run(infos: list[RunInfo]) -> None:
    print("=" * 130)
    print("Per-run runtime")
    print("=" * 130)
    header = (
        f"{'name':<45} {'type':<22} {'progress':>9} {'launches':>9} "
        f"{'active':>9} {'log_wall':>9} {'state_wall':>10}  flags"
    )
    print(header)
    print("-" * len(header))
    for info in sorted(infos, key=lambda r: (r.exp_type, r.model, r.name)):
        progress = f"{info.done_queries}/{info.total_queries}"
        print(
            f"{info.name:<45.45} {info.exp_type:<22.22} "
            f"{progress:>9} {info.n_launches:>9} "
            f"{fmt_dur(info.active_sec):>9} {fmt_dur(info.log_wall_sec):>9} "
            f"{fmt_dur(info.state_wall_sec):>10}  {_flag_str(info)}"
        )


def aggregate(infos: list[RunInfo], key_fn) -> list[tuple[str, dict]]:
    buckets: dict[str, dict] = defaultdict(
        lambda: {
            "n_runs": 0,
            "active_sec": 0.0,
            "log_wall_sec": 0.0,
            "state_wall_sec": 0.0,
            "queries": 0,
            "launches": 0,
        }
    )
    for info in infos:
        b = buckets[key_fn(info)]
        b["n_runs"] += 1
        b["active_sec"] += info.active_sec
        b["log_wall_sec"] += info.log_wall_sec
        b["state_wall_sec"] += info.state_wall_sec
        b["queries"] += info.done_queries
        b["launches"] += info.n_launches
    return sorted(buckets.items())


def print_aggregate(title: str, rows: list[tuple[str, dict]]) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)
    header = (
        f"{'key':<40} {'runs':>5} {'queries':>8} {'launches':>9} "
        f"{'active':>10} {'log_wall':>10} {'state_wall':>11}"
    )
    print(header)
    print("-" * len(header))
    tot_active = tot_log = tot_state = 0.0
    tot_runs = tot_q = tot_lau = 0
    for k, b in rows:
        print(
            f"{k:<40.40} {b['n_runs']:>5} {b['queries']:>8} {b['launches']:>9} "
            f"{fmt_dur(b['active_sec']):>10} {fmt_dur(b['log_wall_sec']):>10} "
            f"{fmt_dur(b['state_wall_sec']):>11}"
        )
        tot_active += b["active_sec"]
        tot_log += b["log_wall_sec"]
        tot_state += b["state_wall_sec"]
        tot_runs += b["n_runs"]
        tot_q += b["queries"]
        tot_lau += b["launches"]
    print("-" * len(header))
    print(
        f"{'TOTAL':<40} {tot_runs:>5} {tot_q:>8} {tot_lau:>9} "
        f"{fmt_dur(tot_active):>10} {fmt_dur(tot_log):>10} {fmt_dur(tot_state):>11}"
    )


def _load_manifest_index(manifest_paths: list[Path]) -> tuple[dict[str, str], set[str]]:
    """Return (slug → 'active'/'disabled', set of declared models).

    'active' wins over 'disabled' if a slug appears in both manifests with
    different statuses.
    """
    try:
        import yaml
    except ImportError:
        return {}, set()
    declared: dict[str, str] = {}
    models: set[str] = set()
    for p in manifest_paths:
        if not p.exists():
            continue
        try:
            data = yaml.safe_load(p.read_text()) or {}
        except Exception:
            continue
        for e in data.get("experiments") or []:
            n = e.get("name")
            if not n:
                continue
            if e.get("model"):
                models.add(e["model"])
            status = "disabled" if e.get("disabled") else "active"
            cur = declared.get(n)
            if cur != "active":
                declared[n] = status
    return declared, models


def _classify(slug: str, model: str, declared: dict[str, str], declared_models: set[str]) -> str:
    if slug in declared:
        return declared[slug]
    if model and model not in declared_models:
        return "stale_model"
    return "stale_run"


def _aggregate_experiments(infos: list[RunInfo]) -> dict[tuple[str, str], dict]:
    """Group by (exp_type, slug); sum log_wall/launches across shards.

    For each (type, slug) group:
      * log_wall_sec / launches / n_runs: summed across all dirs (shards run
        in parallel on separate GPUs, so the sum is the GPU-hour cost; the
        max is the wall-clock latency. We report sum here.)
      * queries: max across the group, since the merged dir already reports
        the union and shards individually report disjoint slices.
      * model: taken from any member (consistent within a slug).
    """
    groups: dict[tuple[str, str], dict] = {}
    for info in infos:
        slug = _experiment_slug(info.name)
        key = (info.exp_type, slug)
        g = groups.setdefault(key, {
            "type": info.exp_type,
            "slug": slug,
            "model": info.model,
            "n_runs": 0,
            "n_shards": 0,
            "queries": 0,
            "launches": 0,
            "log_wall_sec": 0.0,
            "log_wall_max_sec": 0.0,
            "settings": info.settings,
        })
        g["n_runs"] += 1
        if SHARD_SUFFIX_RE.search(info.name):
            g["n_shards"] += 1
        g["queries"] = max(g["queries"], info.done_queries)
        g["launches"] += info.n_launches
        g["log_wall_sec"] += info.log_wall_sec
        g["log_wall_max_sec"] = max(g["log_wall_max_sec"], info.log_wall_sec)
    return groups


def write_json(
    out_path: Path,
    infos: list[RunInfo],
    manifest_paths: Optional[list[Path]] = None,
) -> None:
    """Emit per-model, per-(type, model), and per-experiment log_wall aggregates.

    If `manifest_paths` are provided, each per_experiment entry is tagged with
    a `classification` field (active / disabled / stale_run / stale_model)
    and a `by_classification` rollup is included at the top level.
    """

    def _bucket() -> dict:
        return {
            "n_runs": 0,
            "queries": 0,
            "launches": 0,
            "log_wall_sec": 0.0,
        }

    per_model: dict[str, dict] = defaultdict(_bucket)
    per_type_model: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(_bucket)
    )
    total = _bucket()

    for info in infos:
        for b in (per_model[info.model], per_type_model[info.exp_type][info.model], total):
            b["n_runs"] += 1
            b["queries"] += info.done_queries
            b["launches"] += info.n_launches
            b["log_wall_sec"] += info.log_wall_sec

    def _finalize(b: dict) -> dict:
        return {
            "n_runs": b["n_runs"],
            "queries": b["queries"],
            "launches": b["launches"],
            "log_wall_sec": round(b["log_wall_sec"], 2),
            "log_wall_hours": round(b["log_wall_sec"] / 3600.0, 3),
            "log_wall_human": fmt_dur(b["log_wall_sec"]),
        }

    experiments = _aggregate_experiments(infos)

    declared, declared_models = ({}, set())
    if manifest_paths:
        declared, declared_models = _load_manifest_index(manifest_paths)

    def _finalize_exp(g: dict, cls: Optional[str]) -> dict:
        out = {
            "type": g["type"],
            "model": g["model"],
            "n_runs": g["n_runs"],
            "n_shards": g["n_shards"],
            "queries": g["queries"],
            "launches": g["launches"],
            "log_wall_sec": round(g["log_wall_sec"], 2),
            "log_wall_hours": round(g["log_wall_sec"] / 3600.0, 3),
            "log_wall_human": fmt_dur(g["log_wall_sec"]),
            "log_wall_max_sec": round(g["log_wall_max_sec"], 2),
            "log_wall_max_human": fmt_dur(g["log_wall_max_sec"]),
            "settings": g["settings"],
        }
        if cls is not None:
            out["classification"] = cls
        return out

    sorted_exps = sorted(
        experiments.items(), key=lambda kv: -kv[1]["log_wall_sec"]
    )
    per_experiment_payload: dict[str, dict] = {}
    by_classification: dict[str, dict] = defaultdict(
        lambda: {"n_experiments": 0, "queries": 0, "launches": 0, "log_wall_sec": 0.0}
    )
    for (t, slug), g in sorted_exps:
        cls = _classify(slug, g["model"], declared, declared_models) if declared else None
        per_experiment_payload[f"{t}/{slug}"] = _finalize_exp(g, cls)
        if cls is not None:
            b = by_classification[cls]
            b["n_experiments"] += 1
            b["queries"] += g["queries"]
            b["launches"] += g["launches"]
            b["log_wall_sec"] += g["log_wall_sec"]

    payload = {
        "total": _finalize(total),
        "per_model": {
            m: _finalize(b)
            for m, b in sorted(per_model.items(), key=lambda kv: -kv[1]["log_wall_sec"])
        },
        "per_type_model": {
            t: {m: _finalize(b) for m, b in sorted(models.items())}
            for t, models in sorted(per_type_model.items())
        },
        "per_experiment": per_experiment_payload,
    }
    if by_classification:
        payload["by_classification"] = {
            cls: {
                "n_experiments": b["n_experiments"],
                "queries": b["queries"],
                "launches": b["launches"],
                "log_wall_sec": round(b["log_wall_sec"], 2),
                "log_wall_hours": round(b["log_wall_sec"] / 3600.0, 3),
                "log_wall_human": fmt_dur(b["log_wall_sec"]),
            }
            for cls, b in sorted(
                by_classification.items(),
                key=lambda kv: ("active", "disabled", "stale_run", "stale_model").index(kv[0])
                    if kv[0] in ("active", "disabled", "stale_run", "stale_model") else 99,
            )
        }
        payload["manifests"] = [str(p) for p in manifest_paths]

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def print_per_experiment(infos: list[RunInfo]) -> None:
    """Print a per-experiment table (shards merged into one row by slug)."""
    groups = _aggregate_experiments(infos)
    print()
    print("=" * 130)
    print("Per-experiment runtime (shards summed by slug; queries=max across shards)")
    print("=" * 130)
    header = (
        f"{'experiment':<45} {'type':<22} {'model':<22} "
        f"{'shards':>6} {'queries':>8} {'launches':>9} {'log_wall':>10} {'wall_max':>9}"
    )
    print(header)
    print("-" * len(header))
    tot_wall = 0.0
    for (_t, _slug), g in sorted(
        groups.items(), key=lambda kv: -kv[1]["log_wall_sec"]
    ):
        print(
            f"{g['slug']:<45.45} {g['type']:<22.22} {g['model']:<22.22} "
            f"{g['n_shards']:>6} {g['queries']:>8} {g['launches']:>9} "
            f"{fmt_dur(g['log_wall_sec']):>10} {fmt_dur(g['log_wall_max_sec']):>9}"
        )
        tot_wall += g["log_wall_sec"]
    print("-" * len(header))
    print(
        f"{'TOTAL':<45} {'':<22} {'':<22} {'':>6} {'':>8} {'':>9} "
        f"{fmt_dur(tot_wall):>10} {'':>9}"
    )


def write_csv(out_path: Path, infos: list[RunInfo]) -> None:
    cols = [
        "name", "exp_type", "model", "group", "total_queries", "done_queries",
        "n_launches", "active_sec", "log_wall_sec", "state_wall_sec",
        "is_merged", "is_archived_shard", "has_log", "has_detailed", "run_dir",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for info in sorted(infos, key=lambda r: (r.exp_type, r.model, r.name)):
            w.writerow([
                info.name, info.exp_type, info.model, info.group,
                info.total_queries, info.done_queries, info.n_launches,
                round(info.active_sec, 2), round(info.log_wall_sec, 2),
                round(info.state_wall_sec, 2),
                int(info.is_merged), int(info.is_archived_shard),
                int(info.has_log), int(info.has_detailed),
                str(info.run_dir.relative_to(PROJECT_ROOT) if PROJECT_ROOT in info.run_dir.parents else info.run_dir),
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root", action="append", default=None,
        help=f"Runs root dir; can repeat. Default: {DEFAULT_ROOTS}",
    )
    parser.add_argument(
        "--group-by", choices=("type", "model", "type-model", "none"),
        default="type-model",
        help="Aggregation key for the summary table.",
    )
    parser.add_argument(
        "--no-per-run", action="store_true",
        help="Skip the per-run table; only print the aggregate.",
    )
    parser.add_argument(
        "--per-experiment", action="store_true",
        help="Print a per-experiment table (shards summed by slug).",
    )
    parser.add_argument("--csv", type=Path, help="Write per-run rows to CSV.")
    parser.add_argument(
        "--json", type=Path, dest="json_out",
        help="Write per-model and per-(type, model) log_wall aggregates as JSON.",
    )
    parser.add_argument(
        "--manifest", action="append", default=None,
        help="Manifest YAML path; can repeat. When set, JSON tags each "
             "experiment with active / disabled / stale_run / stale_model "
             f"and adds a `by_classification` block. Default: {DEFAULT_MANIFESTS}",
    )
    parser.add_argument(
        "--include-archived-dups", action="store_true",
        help="Include _postshardprep_/_pre_shard_ archive snapshots that "
             "duplicate live runs (default: skipped).",
    )
    args = parser.parse_args()

    roots = [PROJECT_ROOT / r for r in (args.runs_root or DEFAULT_ROOTS)]
    infos = collect(roots, include_archived_dups=args.include_archived_dups)
    if not infos:
        print("No run directories found under:", ", ".join(str(r) for r in roots))
        sys.exit(1)

    if not args.no_per_run:
        print_per_run(infos)

    if args.per_experiment:
        print_per_experiment(infos)

    key_fns = {
        "type": lambda i: i.exp_type,
        "model": lambda i: i.model,
        "type-model": lambda i: f"{i.exp_type}/{i.model}",
        "none": lambda i: "all",
    }
    rows = aggregate(infos, key_fns[args.group_by])
    print_aggregate(f"Aggregate by {args.group_by}", rows)

    # Always also emit a model-only roll-up at the bottom (it's the most
    # useful slice for "how much GPU time per model").
    if args.group_by != "model":
        rows_model = aggregate(infos, key_fns["model"])
        print_aggregate("Aggregate by model", rows_model)

    if args.csv:
        write_csv(args.csv, infos)
        print(f"\nCSV written: {args.csv}")

    if args.json_out:
        manifest_paths = [
            PROJECT_ROOT / m for m in (args.manifest or DEFAULT_MANIFESTS)
        ]
        write_json(args.json_out, infos, manifest_paths=manifest_paths)
        print(f"JSON written: {args.json_out}")


if __name__ == "__main__":
    main()
