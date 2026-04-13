#!/usr/bin/env python3
"""Compare a fresh detailed_results.jsonl against the frozen pipeline snapshot.

Strict fields must match exactly (set-based for id lists); non-deterministic
fields (timing, list order) are ignored. Prints a violation report and exits
with code 1 if any strict field differs.

Usage:
    python scripts/diff_snapshot.py \\
        code/tests/fixtures/pipeline_snapshot_v0.1.0.jsonl \\
        eval_results/.../detailed_results.jsonl
"""
import json
import sys


def load_first(path):
    with open(path, encoding="utf-8") as f:
        return json.loads(f.readline())


def diff(baseline, current):
    violations = []

    for f in ("idx", "query", "ground_truth_arxiv_ids"):
        if baseline.get(f) != current.get(f):
            violations.append(f"STRICT {f}: {baseline.get(f)!r} != {current.get(f)!r}")

    b_iters = baseline.get("iteration_results", [])
    c_iters = current.get("iteration_results", [])
    if len(b_iters) != len(c_iters):
        violations.append(f"STRICT iter count: {len(b_iters)} != {len(c_iters)}")
        return violations

    for i, (bi, ci) in enumerate(zip(b_iters, c_iters)):
        for list_field in ("current_iter_retrieved", "current_iter_selected"):
            bs = set(bi.get(list_field, []))
            cs = set(ci.get(list_field, []))
            if bs != cs:
                violations.append(
                    f"STRICT iter{i}.{list_field}: "
                    f"missing={bs - cs} extra={cs - bs}"
                )
        if bi.get("iteration_metrics") != ci.get("iteration_metrics"):
            violations.append(f"STRICT iter{i}.iteration_metrics differ")
        if bi.get("subquery_metrics") != ci.get("subquery_metrics"):
            violations.append(f"STRICT iter{i}.subquery_metrics differ")

    bf = {p["arxiv_id"] for p in baseline.get("final_selected_papers", [])}
    cf = {p["arxiv_id"] for p in current.get("final_selected_papers", [])}
    if bf != cf:
        violations.append(
            f"STRICT final_selected: missing={bf - cf} extra={cf - bf}"
        )

    return violations


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    baseline = load_first(sys.argv[1])
    current = load_first(sys.argv[2])
    violations = diff(baseline, current)
    if not violations:
        print("OK — no strict-field differences")
        sys.exit(0)
    print(f"FAIL — {len(violations)} violation(s):")
    for v in violations:
        print(f"  {v}")
    sys.exit(1)


if __name__ == "__main__":
    main()
