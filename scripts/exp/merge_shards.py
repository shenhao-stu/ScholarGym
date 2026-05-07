"""Merge sharded eval run outputs back into a canonical run dir.

After 12 shards (6 nothink + 6 think) finish, run this to combine
detailed_results.jsonl, recompute aggregated metrics, and write the merged
results to runs_iter25/<base-name>/ (without the -shardNofM suffix).

Usage:
    python scripts/exp/merge_shards.py runs_iter25 gemma4-31b-think-none-iter25
    python scripts/exp/merge_shards.py runs_iter25 --pattern 'gemma4-31b-*'
"""
import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARD_RE = re.compile(r"^(.+)-shard(\d+)of(\d+)$")


def find_shard_groups(runs_root: Path, base_filter: str | None = None) -> dict[str, list[Path]]:
    """Return {base_name: [shard_dir, ...]} for completed shard runs."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for d in runs_root.iterdir():
        if not d.is_dir():
            continue
        m = SHARD_RE.match(d.name)
        if not m:
            continue
        base = m.group(1)
        if base_filter and base != base_filter:
            continue
        groups[base].append(d)
    return groups


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _expected_total(shard_dirs: list[Path]) -> int | None:
    """Infer dataset size from BENCHMARK_PATH in any shard's config.py."""
    for d in shard_dirs:
        cfg = d / "config.py"
        if not cfg.exists():
            continue
        for line in cfg.read_text().splitlines():
            if line.strip().startswith("BENCHMARK_PATH"):
                _, _, rhs = line.partition("=")
                rel = rhs.strip().strip("'\"")
                p = (PROJECT_ROOT / rel).resolve()
                if p.exists():
                    return sum(1 for _ in p.open())
    return None


def merge_one(base_name: str, shard_dirs: list[Path], runs_root: Path) -> None:
    shard_dirs = sorted(shard_dirs, key=lambda p: int(SHARD_RE.match(p.name).group(2)))
    n_expected = int(SHARD_RE.match(shard_dirs[0].name).group(3))
    if len(shard_dirs) != n_expected:
        print(f"[warn] {base_name}: found {len(shard_dirs)} of {n_expected} shards")

    out_dir = runs_root / base_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge detailed_results.jsonl (dedup by idx, keep last write)
    by_idx: dict[int, dict] = {}
    for d in shard_dirs:
        for r in load_jsonl(d / "detailed_results.jsonl"):
            idx = r.get("idx")
            if idx is None:
                continue
            by_idx[idx] = r
    merged = [by_idx[i] for i in sorted(by_idx)]
    detailed_out = out_dir / "detailed_results.jsonl"
    with detailed_out.open("w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[merge] {base_name}: {len(merged)} unique queries → {detailed_out}")

    # Aggregate metrics: average all `iter_*` numeric fields per result
    metric_buckets: dict[str, list[float]] = defaultdict(list)
    for r in merged:
        for ir in r.get("iteration_results", []):
            for k, v in ir.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    metric_buckets[f"iter_{ir.get('iter','?')}_{k}"].append(v)

    expected_total = _expected_total(shard_dirs) or len(merged)
    summary = {
        "base_name": base_name,
        "num_shards": n_expected,
        "shards_present": len(shard_dirs),
        "total_queries_expected": expected_total,
        "successful_queries_merged": len(merged),
        "shard_dirs": [d.name for d in shard_dirs],
    }
    # Pick first shard's results.json as template, override counts/metrics
    template_results = None
    for d in shard_dirs:
        rj = list(d.glob("eval_results_*.json"))
        if rj:
            try:
                template_results = json.loads(rj[0].read_text())
                break
            except Exception:
                pass
    if template_results:
        template_results["total_queries"] = expected_total
        template_results["successful_queries"] = len(merged)
        template_results.pop("shard_idx", None)
        template_results.pop("num_shards", None)
        # detailed_results inline list (keep small)
        template_results["detailed_results"] = merged
        # Recompute avg_* metrics from per-iter buckets if any
        for k, vals in metric_buckets.items():
            template_results[f"avg_{k}"] = sum(vals) / len(vals)
        out_results = out_dir / "results_merged.json"
        out_results.write_text(json.dumps(template_results, ensure_ascii=False, indent=2))
        print(f"[merge] {base_name}: results → {out_results}")

    (out_dir / "merge_summary.json").write_text(json.dumps(summary, indent=2))

    # Copy first shard's config.py / manifest.yaml so UI shows model/group
    for src in (shard_dirs[0] / "config.py", shard_dirs[0] / "manifest.yaml"):
        if src.exists() and not (out_dir / src.name).exists():
            shutil.copy2(src, out_dir / src.name)

    # Write a state.json marking the merged dir as 'done'.
    # total_queries reflects the full dataset (incl. invalid samples that the
    # eval pipeline drops upstream); successful_queries = actually merged rows.
    first_state = shard_dirs[0] / "state.json"
    fs = json.loads(first_state.read_text()) if first_state.exists() else {}
    state = {
        "name": base_name,
        "group": fs.get("group", "ungrouped"),
        "type": fs.get("type", "default"),
        "model": fs.get("model") or fs.get("group"),
        "status": "done",
        "pid": None,
        "total_queries": expected_total,
        "successful_queries": len(merged),
        "merged_from": [d.name for d in shard_dirs],
    }
    (out_dir / "state.json").write_text(json.dumps(state, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", help="e.g. runs_iter25")
    ap.add_argument("base_name", nargs="?", default=None, help="Specific base name to merge (omit to merge all complete shard groups)")
    args = ap.parse_args()

    runs_root = (PROJECT_ROOT / args.runs_dir).resolve()
    if not runs_root.exists():
        print(f"[error] {runs_root} does not exist")
        return 1

    groups = find_shard_groups(runs_root, args.base_name)
    if not groups:
        print(f"[info] no shard groups found under {runs_root}" + (f" matching {args.base_name!r}" if args.base_name else ""))
        return 0

    for base, dirs in sorted(groups.items()):
        merge_one(base, dirs, runs_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
