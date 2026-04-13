"""Pipeline snapshot regression test.

Runs `code/eval.py` in LLM cassette replay mode and asserts that the
strict-field set of `detailed_results.jsonl` matches the frozen baseline
at `code/tests/fixtures/pipeline_snapshot_v0.1.0.jsonl`.

Gated by env var `RUN_SNAPSHOT_TEST=1` because it requires:
  - Qdrant running at localhost:6433
  - BM25 index built
  - cassette files present under code/tests/cassettes/

Usage:
    RUN_SNAPSHOT_TEST=1 pytest code/tests/test_pipeline_snapshot.py -v
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = REPO_ROOT / "code" / "tests" / "fixtures" / "pipeline_snapshot_v0.1.0.jsonl"
DIFF_SCRIPT = REPO_ROOT / "scripts" / "diff_snapshot.py"


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("RUN_SNAPSHOT_TEST"),
    reason="Set RUN_SNAPSHOT_TEST=1 to run (requires Qdrant + cassettes)",
)
def test_pipeline_snapshot_replay(tmp_path):
    assert BASELINE.exists(), f"Baseline missing: {BASELINE}"

    # Cleanup previous detailed_results to force a fresh run (no checkpoint skip)
    eval_dir = (
        REPO_ROOT
        / "eval_results"
        / "qwen3-8b_complex_vector_deep_research_maxq-10_iter-2_reasoning_non-structured_NONE"
    )
    detailed = eval_dir / "detailed_results.jsonl"
    if detailed.exists():
        detailed.unlink()

    env = os.environ.copy()
    env["LLM_CASSETTE_MODE"] = "replay"
    env["no_proxy"] = "localhost,127.0.0.1"

    result = subprocess.run(
        [
            sys.executable, "code/eval.py",
            "--benchmark_jsonl", "data/scholargym_bench_1q.jsonl",
            "--is_local", "--llm_model", "qwen3-8b",
            "--workflow", "deep_research",
            "--search_method", "vector",
            "--max_iterations", "2",
            "--browser_mode", "NONE",
        ],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"eval.py failed: {result.stderr[-2000:]}"
    assert detailed.exists(), f"detailed_results.jsonl not created at {detailed}"

    diff_result = subprocess.run(
        [sys.executable, str(DIFF_SCRIPT), str(BASELINE), str(detailed)],
        capture_output=True, text=True,
    )
    assert diff_result.returncode == 0, (
        f"Snapshot drift detected:\n{diff_result.stdout}\n{diff_result.stderr}"
    )
