"""Split evaluation_summary.jsonl into fast/hard/full subsets with **full-set denominator**.

Differences from paper_metrics/summary_devider.py (do NOT modify that file):
  * Denominator for averaged metrics is the *target subset size* (fast=200,
    hard=100, full=300), not `len(values)`. Queries that never produced a
    result are treated as 0 — matching eval.py's avg formula and ICML Table 1
    convention.
  * Reads ONLY runs_iter25/evaluation_summary.jsonl (and the detailed_results
    paths it points at). Writes outputs next to the input.

Usage:
    python scripts/exp/summary_split_full_denom.py \
        runs_iter25/evaluation_summary.jsonl
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Set
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRUTH_FAST = PROJECT_ROOT / 'data/test_fast.jsonl'
TRUTH_HARD = PROJECT_ROOT / 'data/test_hard.jsonl'

FAST_TOTAL = 200
HARD_TOTAL = 100
FULL_TOTAL = 300

# Metrics where missing query counts as 0 (averaged over target_total queries)
QUERY_AVG_PREFIXES = (
    'recall_iter_', 'precision_iter_', 'retrieval_recall_iter_',
    'retrieval_precision_iter_', 'missed_gt_ratio_iter_',
    'retrieved_count_iter_', 'selected_count_iter_',
    'avg_distance_iter_', 'discarded_ratio_iter_',
    'discarded_total_count_iter_',
    'cur_retrieved_iter_', 'cur_selected_iter_',
)
# Phase timings: leave as np.mean over actual observations (a missing query
# shouldn't push a 0 timing into the average).
PHASE_TIMING_KEYS = {'planner_during', 'retrieval_during', 'selector_during',
                     'browser_during', 'overhead_during', 'total_during'}


def _is_query_metric(key: str) -> bool:
    return any(key.startswith(p) for p in QUERY_AVG_PREFIXES)


def _load_qid_set(path: Path) -> Set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    out = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = r.get('qid')
            if qid:
                out.add(qid)
    return out


_BENCH_CACHE: Dict[str, Dict[int, str]] = {}

def _bench_idx_to_qid(bench_path: Path) -> Dict[int, str]:
    """Cache idx → qid map for a benchmark file."""
    key = str(bench_path)
    if key in _BENCH_CACHE:
        return _BENCH_CACHE[key]
    m: Dict[int, str] = {}
    if bench_path.exists():
        with bench_path.open() as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                qid = r.get('qid')
                if qid:
                    m[i] = qid
    _BENCH_CACHE[key] = m
    return m


def _read_bench_path_from_config(run_dir: Path) -> Optional[Path]:
    cfg = run_dir / 'config.py'
    if not cfg.exists():
        return None
    for line in cfg.read_text().splitlines():
        s = line.strip()
        if s.startswith('BENCHMARK_PATH'):
            _, _, rhs = s.partition('=')
            rel = rhs.strip().strip("'\"")
            return (PROJECT_ROOT / rel).resolve()
    return None


def build_filter_idxs(run_dir: Path, fast_qids: Set[str],
                      hard_qids: Set[str]) -> tuple[Set[int], Set[int]]:
    """Compute fast/hard idx sets in the run's specific benchmark space.

    Each run may use a different BENCHMARK_PATH; the idxs in
    detailed_results.jsonl reference *that* benchmark's positions, so we
    must map idx→qid for that benchmark and check qid against the canonical
    test_fast/test_hard truth.
    """
    bench = _read_bench_path_from_config(run_dir)
    if bench is None:
        return set(), set()
    idx_to_qid = _bench_idx_to_qid(bench)
    fast_idxs = {i for i, q in idx_to_qid.items() if q in fast_qids}
    hard_idxs = {i for i, q in idx_to_qid.items() if q in hard_qids}
    return fast_idxs, hard_idxs


def _process_detailed_file(path: Path, max_iterations: int,
                           filter_idxs: Optional[Set[int]],
                           target_total: int) -> Dict[str, Any]:
    """Aggregate metrics from detailed_results.jsonl, restricted to filter_idxs.

    target_total is the size of the target subset (fast=200, hard=100, full=300).
    Per-query metrics are averaged as sum / target_total (missing queries → 0).
    """
    metric_lists: Dict[str, list] = {}
    metric_names = ['recall', 'precision', 'retrieval_recall',
                    'retrieval_precision', 'missed_gt_ratio',
                    'retrieved_count', 'selected_count']

    if not path.exists():
        return {}

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                q = json.loads(line)
            except json.JSONDecodeError:
                continue
            q_idx = q.get('idx')
            if q_idx is None:
                continue
            if filter_idxs is not None and q_idx not in filter_idxs:
                continue

            iter_results = q.get('iteration_results', [])
            if not iter_results:
                continue

            # collect per-iter values for this query
            query_metrics_by_iter = {m: {} for m in metric_names}
            for res in iter_results:
                it = res.get('iter_idx')
                if it is None:
                    continue
                for m in metric_names:
                    if m in res:
                        query_metrics_by_iter[m][it] = res[m]
                # phase timings (kept as mean over actual data)
                for p in PHASE_TIMING_KEYS:
                    v = res.get(p, -1)
                    if v >= 0:
                        metric_lists.setdefault(p, []).append(v)
                # avg_distance per iter
                d = res.get('avg_distance', -1)
                if d >= 0:
                    metric_lists.setdefault(f'avg_distance_iter_{it}', []).append(d)
                # discarded per iter
                im = res.get('iteration_metrics', {})
                for sub in ['discarded_ratio', 'discarded_total_count']:
                    v = im.get(sub, -1)
                    if v >= 0:
                        metric_lists.setdefault(f'{sub}_iter_{it}', []).append(v)
                # current-iter (NOT cumulative) retrieved/selected counts:
                # iteration_metrics.retrieval.total = # of papers this iter's
                # subqueries returned after cross-subquery dedup.
                # Differs from `retrieved_count` in iter_results which is
                # cumulative across iters.
                ret_total = im.get('retrieval', {}).get('total')
                sel_total = im.get('selection', {}).get('total')
                if ret_total is not None:
                    metric_lists.setdefault(f'cur_retrieved_iter_{it}', []).append(ret_total)
                if sel_total is not None:
                    metric_lists.setdefault(f'cur_selected_iter_{it}', []).append(sel_total)

            # smoothing: forward-fill missing iters with last seen value
            for m, iters in query_metrics_by_iter.items():
                if not iters:
                    continue
                last_val = iters[max(iters.keys())]
                for it in range(1, max_iterations + 1):
                    val = iters.get(it, last_val)
                    metric_lists.setdefault(f'{m}_iter_{it}', []).append(val)

    # aggregate
    final = {}
    for k, vals in metric_lists.items():
        if not vals:
            continue
        if k in PHASE_TIMING_KEYS:
            final[f'avg_{k}'] = float(np.mean(vals))
        elif _is_query_metric(k):
            # full-set denominator
            final[f'avg_{k}'] = sum(vals) / target_total if target_total else float(np.mean(vals))
        else:
            final[f'avg_{k}'] = float(np.mean(vals))

    # macro avg of per-iter missed_gt_ratio
    macro = [v for k, v in final.items() if k.startswith('avg_missed_gt_ratio_iter_')]
    if macro:
        final['avg_missed_gt_ratio_macro_avg'] = float(np.mean(macro))
    return final


def _construct_record(old: Dict, metrics: Dict, max_iters: int,
                       subset_label: str, target_total: int) -> Dict:
    res = {}
    base_keys = [
        'model_name', 'prompt_type', 'search_method', 'workflow',
        'enable_reasoning', 'enable_structured_output', 'EVAL_TOP_K_VALUES',
        'MAX_RESULTS_PER_QUERY', 'EVAL_MAX_ITERATIONS',
        'EVAL_DETAILED_RESULTS_PATH', 'GT_RANK_CUTOFF', 'BROWSER_MODE',
        'PLANNER_ABLATION',
    ]
    for k in base_keys:
        if k in old:
            res[k] = old[k]
    res['_subset'] = subset_label
    res['_subset_total'] = target_total

    cleaned = {k.replace('avg_', ''): v for k, v in metrics.items()}
    groups = ['distance', 'recall', 'precision', 'retrieval_recall',
              'retrieval_precision', 'missed_gt_ratio', 'retrieved_count',
              'selected_count', 'cur_retrieved', 'cur_selected']
    for g in groups:
        for it in range(1, max_iters + 1):
            k = f'{g}_iter_{it}'
            if k in cleaned:
                res[k] = cleaned[k]

    for t in PHASE_TIMING_KEYS:
        if t in cleaned:
            res[t] = cleaned[t]

    for it in range(1, max_iters + 1):
        for sub in ['discarded_ratio', 'discarded_total_count']:
            k = f'{sub}_iter_{it}'
            if k in cleaned:
                res[k] = cleaned[k]

    if 'missed_gt_ratio_macro_avg' in cleaned:
        res['missed_gt_ratio_macro_avg'] = cleaned['missed_gt_ratio_macro_avg']
    return res


def split(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    fast_qids = _load_qid_set(TRUTH_FAST)
    hard_qids = _load_qid_set(TRUTH_HARD)
    print(f'Loaded canonical truth qids: fast={len(fast_qids)} (expected {FAST_TOTAL}), '
          f'hard={len(hard_qids)} (expected {HARD_TOTAL})')

    out_full = input_path.with_name(input_path.stem + '_full.jsonl')
    out_fast = input_path.with_name(input_path.stem + '_fast.jsonl')
    out_hard = input_path.with_name(input_path.stem + '_hard.jsonl')
    for p in (out_full, out_fast, out_hard):
        if p.exists():
            p.unlink()

    seen = 0
    written_full = written_fast = written_hard = 0

    with input_path.open() as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                old = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen += 1
            detailed_rel = old.get('EVAL_DETAILED_RESULTS_PATH')
            if not detailed_rel:
                continue
            detailed = (PROJECT_ROOT / detailed_rel).resolve()
            if not detailed.exists():
                # may also be already absolute
                if Path(detailed_rel).exists():
                    detailed = Path(detailed_rel)
                else:
                    print(f'  skip (no detailed file): {detailed_rel}')
                    continue
            max_iter = int(old.get('EVAL_MAX_ITERATIONS', 5) or 5)

            # Each run may use a different benchmark file; compute fast/hard
            # idx sets in *that* benchmark's index space.
            run_dir = detailed.parent
            fast_idxs, hard_idxs = build_filter_idxs(run_dir, fast_qids, hard_qids)

            full_m = _process_detailed_file(detailed, max_iter, None, FULL_TOTAL)
            fast_m = _process_detailed_file(detailed, max_iter, fast_idxs, FAST_TOTAL)
            hard_m = _process_detailed_file(detailed, max_iter, hard_idxs, HARD_TOTAL)

            if full_m:
                with out_full.open('a') as g:
                    g.write(json.dumps(_construct_record(old, full_m, max_iter, 'full', FULL_TOTAL),
                                        ensure_ascii=False) + '\n')
                written_full += 1
            if fast_m:
                with out_fast.open('a') as g:
                    g.write(json.dumps(_construct_record(old, fast_m, max_iter, 'fast', FAST_TOTAL),
                                        ensure_ascii=False) + '\n')
                written_fast += 1
            if hard_m:
                with out_hard.open('a') as g:
                    g.write(json.dumps(_construct_record(old, hard_m, max_iter, 'hard', HARD_TOTAL),
                                        ensure_ascii=False) + '\n')
                written_hard += 1

    print(f'\n[done] read {seen} summary rows;\n'
          f'  full → {out_full.name} ({written_full})\n'
          f'  fast → {out_fast.name} ({written_fast})\n'
          f'  hard → {out_hard.name} ({written_hard})')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: summary_split_full_denom.py <evaluation_summary.jsonl>')
        sys.exit(1)
    split(Path(sys.argv[1]).resolve())
