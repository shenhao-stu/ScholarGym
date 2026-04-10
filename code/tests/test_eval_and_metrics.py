"""
Tests for ScholarGym eval workflow details and metrics calculation.

Covers:
- MetricsCalculator: subquery_metrics, iteration_metrics, gt_rank_and_distance, simple_avg_distance
- calculate_retrieval_metrics (utils.py)
- extract_ground_truth_arxiv_ids, parse_json_from_tag, remove_think_blocks (utils.py)
- CheckpointManager (utils.py)
- CitationEvaluator: evaluate_single_query_deep_research, evaluate_benchmark aggregation (eval.py)

Test data is loaded from code/tests/fixtures/*.json via conftest.py fixtures.
"""
import json
import os
import types
import pytest
from unittest.mock import MagicMock, patch
from conftest import load_fixture

# conftest.py already sets sys.path and injects a minimal config module
import config
from structures import Paper, SubQuery, SubQueryState
from metrics import MetricsCalculator
from utils import (
    calculate_retrieval_metrics,
    extract_ground_truth_arxiv_ids,
    parse_json_from_tag,
    remove_think_blocks,
    CheckpointManager,
)


# ---------------------------------------------------------------------------
# Load parametrize data from fixture files
# ---------------------------------------------------------------------------
_metrics_data = load_fixture("metrics_cases.json")
_utils_data = load_fixture("utils_cases.json")


def _param_id(case):
    return case["id"]


# =============================================================================
# 1. MetricsCalculator.calculate_subquery_metrics
# =============================================================================
class TestCalculateSubqueryMetrics:
    """Tests for MetricsCalculator.calculate_subquery_metrics."""

    @pytest.mark.parametrize("case", _metrics_data["subquery_metrics"], ids=_param_id)
    def test_subquery_metrics(self, case):
        retrieved = set(case["retrieved"])
        selected = set(case["selected"])
        gt = set(case["gt"])
        expected = case["expected"]

        result = MetricsCalculator.calculate_subquery_metrics(retrieved, selected, gt)

        assert result["retrieval"]["recall"] == pytest.approx(expected["ret_recall"], abs=1e-3)
        assert result["retrieval"]["precision"] == pytest.approx(expected["ret_prec"], abs=1e-3)
        assert result["selection"]["recall"] == pytest.approx(expected["sel_recall"], abs=1e-3)
        assert result["selection"]["precision"] == pytest.approx(expected["sel_prec"], abs=1e-3)
        assert result["discarded_gt_count"] == expected["disc_count"]

        if "disc_ids" in expected:
            assert set(result["discarded_gt_ids"]) == set(expected["disc_ids"])


# =============================================================================
# 2. MetricsCalculator.calculate_iteration_metrics
# =============================================================================
class TestCalculateIterationMetrics:
    """Tests for MetricsCalculator.calculate_iteration_metrics."""

    @pytest.mark.parametrize("case", _metrics_data["iteration_metrics"], ids=_param_id)
    def test_iteration_metrics(self, case):
        # Convert JSON keys (strings) to int keys, lists to sets
        subquery_results = {}
        for k, v in case["subquery_results"].items():
            subquery_results[int(k)] = {
                "retrieved_arxiv_ids": set(v["retrieved_arxiv_ids"]),
                "selected_arxiv_ids": set(v["selected_arxiv_ids"]),
            }
        gt = set(case["gt"])
        expected = case["expected"]

        result = MetricsCalculator.calculate_iteration_metrics(subquery_results, gt)

        if "ret_recall" in expected:
            assert result["retrieval"]["recall"] == pytest.approx(expected["ret_recall"], abs=1e-3)
        if "ret_matched" in expected:
            assert result["retrieval"]["matched"] == expected["ret_matched"]
        if "ret_total" in expected:
            assert result["retrieval"]["total"] == expected["ret_total"]
        if "sel_recall" in expected:
            assert result["selection"]["recall"] == pytest.approx(expected["sel_recall"], abs=1e-3)
        if "sel_matched" in expected:
            assert result["selection"]["matched"] == expected["sel_matched"]
        if "sel_total" in expected:
            assert result["selection"]["total"] == expected["sel_total"]
        if "ret_f1" in expected:
            assert result["retrieval"]["f1"] == pytest.approx(expected["ret_f1"], abs=1e-3)
        if "sel_f1" in expected:
            assert result["selection"]["f1"] == pytest.approx(expected["sel_f1"], abs=1e-3)
        if "disc_count" in expected:
            assert result["discarded_gt_count"] == expected["disc_count"]
        if "disc_total" in expected:
            assert result["discarded_total_count"] == expected["disc_total"]
        if "disc_ratio" in expected:
            assert result["discarded_ratio"] == pytest.approx(expected["disc_ratio"], abs=1e-3)


# =============================================================================
# 3. MetricsCalculator.calculate_gt_rank_and_distance
# =============================================================================
class TestCalculateGtRankAndDistance:
    """Tests for MetricsCalculator.calculate_gt_rank_and_distance."""

    def test_basic_single_subquery(self, gt_ids_a):
        subqueries = {1: SubQuery(id=1, text="test query")}
        rank_dicts = {1: {"gt1": {"rank": 5, "total": 100}, "gt2": {"rank": 20, "total": 100}}}
        selected_tracker = set()
        min_rank_tracker = {}

        result = MetricsCalculator.calculate_gt_rank_and_distance(
            subqueries, rank_dicts, gt_ids_a, selected_tracker, min_rank_tracker, gt_rank_cutoff=100
        )

        assert result["cur_iter_distances"]["gt1"] == pytest.approx(0.95)
        assert result["cur_iter_distances"]["gt2"] == pytest.approx(0.80)
        assert result["cur_iter_distances"]["gt3"] == pytest.approx(0.0)
        assert result["avg_distance"] == pytest.approx((0.95 + 0.80 + 0.0) / 3)

    def test_multi_subquery_min_rank(self, gt_ids_a):
        subqueries = {
            1: SubQuery(id=1, text="q1"),
            2: SubQuery(id=2, text="q2"),
        }
        rank_dicts = {
            1: {"gt1": {"rank": 10, "total": 100}},
            2: {"gt1": {"rank": 3, "total": 100}},
        }

        result = MetricsCalculator.calculate_gt_rank_and_distance(
            subqueries, rank_dicts, gt_ids_a, set(), {}, gt_rank_cutoff=100
        )

        assert result["cur_iter_distances"]["gt1"] == pytest.approx(1 - 3 / 100)
        sq1_ranks = {r["arxiv_id"]: r["rank"] for r in result["gt_rank"][0]["ranks"]}
        sq2_ranks = {r["arxiv_id"]: r["rank"] for r in result["gt_rank"][1]["ranks"]}
        assert sq1_ranks["gt1"] == 10
        assert sq2_ranks["gt1"] == 3

    def test_historical_rank_tracker(self, gt_ids_a):
        subqueries = {1: SubQuery(id=1, text="q1")}
        rank_dicts = {1: {"gt1": {"rank": 50, "total": 100}}}
        selected_tracker = {"gt1"}
        min_rank_tracker = {"gt1": 5}

        result = MetricsCalculator.calculate_gt_rank_and_distance(
            subqueries, rank_dicts, gt_ids_a, selected_tracker, min_rank_tracker, gt_rank_cutoff=100
        )

        assert result["cur_iter_distances"]["gt1"] == pytest.approx(1 - 5 / 100)
        assert result["updated_selected_min_rank_tracker"]["gt1"] == 5

    def test_distance_clamped_to_zero(self, gt_ids_a):
        subqueries = {1: SubQuery(id=1, text="q1")}
        rank_dicts = {1: {"gt1": {"rank": 200, "total": 500}}}

        result = MetricsCalculator.calculate_gt_rank_and_distance(
            subqueries, rank_dicts, gt_ids_a, set(), {}, gt_rank_cutoff=100
        )
        assert result["cur_iter_distances"]["gt1"] == pytest.approx(0.0)

    def test_gt_not_found_anywhere(self, gt_ids_a):
        subqueries = {1: SubQuery(id=1, text="q1")}
        rank_dicts = {1: {}}

        result = MetricsCalculator.calculate_gt_rank_and_distance(
            subqueries, rank_dicts, gt_ids_a, set(), {}, gt_rank_cutoff=100
        )
        for gid in gt_ids_a:
            assert result["cur_iter_distances"][gid] == pytest.approx(0.0)
        assert result["avg_distance"] == pytest.approx(0.0)

    def test_updated_min_rank_tracker(self, gt_ids_a):
        subqueries = {1: SubQuery(id=1, text="q1")}
        rank_dicts = {1: {"gt1": {"rank": 8, "total": 100}, "gt2": {"rank": 15, "total": 100}}}
        min_rank_tracker = {"gt1": 12}

        result = MetricsCalculator.calculate_gt_rank_and_distance(
            subqueries, rank_dicts, gt_ids_a, {"gt1"}, min_rank_tracker, gt_rank_cutoff=100
        )

        updated = result["updated_selected_min_rank_tracker"]
        assert updated["gt1"] == 8
        assert updated["gt2"] == 15

    def test_empty_gt_arxiv_ids(self):
        subqueries = {1: SubQuery(id=1, text="q1")}
        rank_dicts = {1: {"gt1": {"rank": 5, "total": 100}}}

        result = MetricsCalculator.calculate_gt_rank_and_distance(
            subqueries, rank_dicts, set(), set(), {}, gt_rank_cutoff=100
        )
        assert result["avg_distance"] == pytest.approx(0.0)
        assert result["cur_iter_distances"] == {}

    def test_single_gt_found_at_rank_1(self, gt_ids_single):
        subqueries = {1: SubQuery(id=1, text="q1")}
        rank_dicts = {1: {"gt1": {"rank": 1, "total": 100}}}

        result = MetricsCalculator.calculate_gt_rank_and_distance(
            subqueries, rank_dicts, gt_ids_single, set(), {}, gt_rank_cutoff=100
        )
        assert result["cur_iter_distances"]["gt1"] == pytest.approx(0.99)
        assert result["avg_distance"] == pytest.approx(0.99)

    def test_large_gt_set_partial_found(self, gt_ids_large):
        subqueries = {1: SubQuery(id=1, text="q1")}
        rank_dicts = {1: {
            "gt1": {"rank": 1, "total": 100},
            "gt5": {"rank": 10, "total": 100},
        }}

        result = MetricsCalculator.calculate_gt_rank_and_distance(
            subqueries, rank_dicts, gt_ids_large, set(), {}, gt_rank_cutoff=100
        )
        assert result["cur_iter_distances"]["gt1"] == pytest.approx(0.99)
        assert result["cur_iter_distances"]["gt5"] == pytest.approx(0.90)
        # 8 unfound GTs get distance 0
        assert result["avg_distance"] == pytest.approx((0.99 + 0.90) / 10)


# =============================================================================
# 4. MetricsCalculator.calculate_simple_avg_distance
# =============================================================================
class TestCalculateSimpleAvgDistance:
    """Tests for MetricsCalculator.calculate_simple_avg_distance."""

    @pytest.mark.parametrize("case", _metrics_data["simple_avg_distance"], ids=_param_id)
    def test_simple_avg_distance(self, case):
        gt = set(case["gt"])
        result = MetricsCalculator.calculate_simple_avg_distance(
            case["rank_dicts"], gt, case["cutoff"]
        )
        assert result == pytest.approx(case["expected"], abs=1e-3)


# =============================================================================
# 5. calculate_retrieval_metrics (utils.py)
# =============================================================================
class TestCalculateRetrievalMetrics:
    """Tests for the unified calculate_retrieval_metrics function."""

    @pytest.mark.parametrize("case", _metrics_data["retrieval_metrics"], ids=_param_id)
    def test_retrieval_metrics(self, case):
        gt = set(case["gt"])
        retrieved = set(case["retrieved"])
        selected = set(case["selected"]) if case["selected"] is not None else None
        expected = case["expected"]

        if selected is not None:
            metrics = calculate_retrieval_metrics(gt, retrieved, selected)
        else:
            metrics = calculate_retrieval_metrics(gt, retrieved)

        for key, val in expected.items():
            assert metrics[key] == pytest.approx(val, abs=1e-3), f"Mismatch on {key}"


# =============================================================================
# 6. Utility functions (utils.py)
# =============================================================================
class TestExtractGroundTruthArxivIds:
    """Tests for extract_ground_truth_arxiv_ids."""

    @pytest.mark.parametrize("case", _utils_data["extract_ground_truth"], ids=_param_id)
    def test_extract(self, case):
        result = extract_ground_truth_arxiv_ids(case["papers"], case["labels"])
        assert result == set(case["expected"])


class TestParseJsonFromTag:
    """Tests for parse_json_from_tag."""

    @pytest.mark.parametrize("case", _utils_data["parse_json_from_tag"], ids=_param_id)
    def test_parse(self, case):
        result = parse_json_from_tag(case["response"], case["tag"])
        assert result == case["expected"]


class TestRemoveThinkBlocks:
    """Tests for remove_think_blocks."""

    @pytest.mark.parametrize("case", _utils_data["remove_think_blocks"], ids=_param_id)
    def test_remove(self, case):
        assert remove_think_blocks(case["input"]) == case["expected"]


# =============================================================================
# 7. CheckpointManager (utils.py)
# =============================================================================
class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_append_and_is_processed(self, tmp_path):
        cp_file = str(tmp_path / "checkpoint.jsonl")
        cm = CheckpointManager(cp_file)
        cm.append_result({"idx": 0, "query": "test"})
        assert cm.is_processed(0)
        assert not cm.is_processed(1)

    def test_load_checkpoint(self, tmp_path):
        cp_file = str(tmp_path / "checkpoint.jsonl")
        with open(cp_file, "w") as f:
            f.write(json.dumps({"idx": 0, "query": "q0"}) + "\n")
            f.write(json.dumps({"idx": 2, "query": "q2"}) + "\n")

        cm = CheckpointManager(cp_file)
        processed, cached = cm.load_checkpoint()
        assert processed == {0, 2}
        assert len(cached) == 2

    def test_load_nonexistent(self, tmp_path):
        cp_file = str(tmp_path / "no_such_file.jsonl")
        cm = CheckpointManager(cp_file)
        processed, cached = cm.load_checkpoint()
        assert processed == set()
        assert cached == []

    @pytest.mark.parametrize("case", _utils_data["checkpoint_rebuild_deep_research"], ids=_param_id)
    def test_rebuild_deep_research_stats(self, case, tmp_path):
        cp_file = str(tmp_path / "checkpoint.jsonl")
        cm = CheckpointManager(cp_file)
        cm.cached_results = [case["cached_result"]]

        results = {
            "total_queries": 1,
            "successful_queries": 0,
            "detailed_results": [],
        }
        cm.rebuild_statistics(results, workflow="deep_research", max_iterations=case["max_iterations"])

        for key, val in case["expected"].items():
            assert results[key] == val, f"Mismatch on {key}: {results[key]} != {val}"

    @pytest.mark.parametrize("case", _utils_data["checkpoint_rebuild_simple"], ids=_param_id)
    def test_rebuild_simple_stats(self, case, tmp_path):
        cp_file = str(tmp_path / "checkpoint.jsonl")
        cm = CheckpointManager(cp_file)
        cm.cached_results = [case["cached_result"]]

        results = {
            "total_queries": 1,
            "successful_queries": 0,
            "detailed_results": [],
        }
        for k in case["top_k_list"]:
            results[f"recall@{k}"] = []
            results[f"precision@{k}"] = []

        cm.rebuild_statistics(results, workflow="simple", top_k_list=case["top_k_list"])

        for key, val in case["expected"].items():
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], float):
                for i, v in enumerate(val):
                    assert results[key][i] == pytest.approx(v, abs=1e-3), f"Mismatch on {key}[{i}]"
            else:
                assert results[key] == val, f"Mismatch on {key}"

    def test_multiple_appends(self, tmp_path):
        cp_file = str(tmp_path / "checkpoint.jsonl")
        cm = CheckpointManager(cp_file)
        for i in range(5):
            cm.append_result({"idx": i, "query": f"q{i}"})
        for i in range(5):
            assert cm.is_processed(i)
        assert not cm.is_processed(5)

    def test_load_preserves_order(self, tmp_path):
        cp_file = str(tmp_path / "checkpoint.jsonl")
        with open(cp_file, "w") as f:
            for i in [3, 1, 4, 1, 5]:
                f.write(json.dumps({"idx": i, "query": f"q{i}"}) + "\n")

        cm = CheckpointManager(cp_file)
        processed, cached = cm.load_checkpoint()
        assert processed == {1, 3, 4, 5}
        assert len(cached) == 5


# =============================================================================
# 8. CitationEvaluator - evaluate_single_query_deep_research (eval.py)
# =============================================================================
class TestEvaluateSingleQueryDeepResearch:
    """Tests for CitationEvaluator.evaluate_single_query_deep_research with mocked workflow."""

    def _make_mock_workflow_result(self, iterations):
        """Build a mock workflow result with the given iteration history."""
        history = []
        for it in iterations:
            history.append({
                "stage": "select",
                "iter_idx": it["iter_idx"],
                "retrieved_papers": it.get("retrieved_papers", {}),
                "selected_papers": it.get("selected_papers", {}),
                "gt_rank": it.get("gt_rank", []),
                "browsing_arxiv_ids": it.get("browsing_arxiv_ids", []),
                "avg_distance": it.get("avg_distance", 0),
                "iteration_metrics": it.get("iteration_metrics", {}),
                "subquery_metrics": it.get("subquery_metrics", {}),
                "planner_during": 1.0,
                "retrieval_during": 2.0,
                "selector_during": 3.0,
                "browser_during": 0.0,
                "overhead_during": 0.5,
                "total_during": 6.5,
            })

        return {
            "history": history,
            "selected_papers": [Paper(id="p1", title="Paper 1", abstract="", arxiv_id="gt1")],
            "final_report": "",
            "executed_queries": ["test query"],
        }

    def _make_evaluator(self):
        """Create a CitationEvaluator with mocked workflows."""
        import eval as eval_module
        mock_rag = MagicMock()
        mock_rag.get_available_search_methods.return_value = ["bm25"]

        eval_module.DeepResearchWorkflow = MagicMock
        eval_module.SimpleWorkflow = MagicMock

        from eval import CitationEvaluator
        evaluator = CitationEvaluator(
            rag_system=mock_rag
        )
        return evaluator

    def test_cumulative_metrics(self, sample_benchmark_data):
        evaluator = self._make_evaluator()

        mock_result = self._make_mock_workflow_result([
            {
                "iter_idx": 1,
                "retrieved_papers": {1: [{"arxiv_id": "2009.02040"}, {"arxiv_id": "noise1"}]},
                "selected_papers": {1: [{"arxiv_id": "2009.02040"}]},
                "iteration_metrics": {"discarded_gt_count": 0},
                "avg_distance": 0.9,
            },
            {
                "iter_idx": 2,
                "retrieved_papers": {1: [{"arxiv_id": "1901.00137"}, {"arxiv_id": "noise2"}]},
                "selected_papers": {1: [{"arxiv_id": "1901.00137"}]},
                "iteration_metrics": {"discarded_gt_count": 0},
                "avg_distance": 0.8,
            },
        ])
        evaluator.deep_research_workflow = MagicMock()
        evaluator.deep_research_workflow.run.return_value = mock_result

        query_data = sample_benchmark_data[0]  # GT = {"2009.02040", "1901.00137"}
        result = evaluator.evaluate_single_query_deep_research(
            query_data, idx=0
        )

        assert result is not None
        assert len(result["iteration_results"]) == 2

        iter1 = result["iteration_results"][0]
        assert iter1["iter_idx"] == 1
        assert "2009.02040" in iter1["current_iter_retrieved"]

        iter2 = result["iteration_results"][1]
        assert iter2["recall"] == pytest.approx(1.0)

    def test_null_workflow_result(self, sample_benchmark_data):
        evaluator = self._make_evaluator()
        evaluator.deep_research_workflow = MagicMock()
        evaluator.deep_research_workflow.run.return_value = None

        query_data = sample_benchmark_data[0]
        result = evaluator.evaluate_single_query_deep_research(query_data, idx=0)
        assert result is None

    def test_discarded_gt_accumulation(self, sample_benchmark_data):
        evaluator = self._make_evaluator()

        mock_result = self._make_mock_workflow_result([
            {
                "iter_idx": 1,
                "retrieved_papers": {1: [{"arxiv_id": "2009.02040"}, {"arxiv_id": "1901.00137"}]},
                "selected_papers": {1: [{"arxiv_id": "2009.02040"}]},
                "iteration_metrics": {"discarded_gt_count": 1},
            },
            {
                "iter_idx": 2,
                "retrieved_papers": {1: [{"arxiv_id": "1901.00137"}]},
                "selected_papers": {1: [{"arxiv_id": "1901.00137"}]},
                "iteration_metrics": {"discarded_gt_count": 0},
            },
        ])
        evaluator.deep_research_workflow = MagicMock()
        evaluator.deep_research_workflow.run.return_value = mock_result

        result = evaluator.evaluate_single_query_deep_research(
            sample_benchmark_data[0], idx=0
        )

        assert result["iteration_results"][0]["total_discarded_gt_count"] == 1
        assert result["iteration_results"][1]["total_discarded_gt_count"] == 1

    def test_single_iteration_single_gt(self, sample_benchmark_data):
        """Test with single ground truth paper (entry index 3)."""
        evaluator = self._make_evaluator()

        mock_result = self._make_mock_workflow_result([
            {
                "iter_idx": 1,
                "retrieved_papers": {1: [{"arxiv_id": "2101.00001"}]},
                "selected_papers": {1: [{"arxiv_id": "2101.00001"}]},
                "iteration_metrics": {"discarded_gt_count": 0},
                "avg_distance": 0.95,
            },
        ])
        evaluator.deep_research_workflow = MagicMock()
        evaluator.deep_research_workflow.run.return_value = mock_result

        query_data = sample_benchmark_data[3]  # single_gt entry
        result = evaluator.evaluate_single_query_deep_research(
            query_data, idx=3
        )

        assert result is not None
        assert result["iteration_results"][0]["recall"] == pytest.approx(1.0)

    def test_three_iterations_progressive(self, sample_benchmark_data):
        """Test progressive recall improvement over 3 iterations."""
        evaluator = self._make_evaluator()

        mock_result = self._make_mock_workflow_result([
            {
                "iter_idx": 1,
                "retrieved_papers": {1: [{"arxiv_id": "1706.03762"}, {"arxiv_id": "noise1"}]},
                "selected_papers": {1: [{"arxiv_id": "1706.03762"}]},
                "iteration_metrics": {"discarded_gt_count": 0},
            },
            {
                "iter_idx": 2,
                "retrieved_papers": {1: [{"arxiv_id": "1810.04805"}]},
                "selected_papers": {1: [{"arxiv_id": "1810.04805"}]},
                "iteration_metrics": {"discarded_gt_count": 0},
            },
            {
                "iter_idx": 3,
                "retrieved_papers": {1: [{"arxiv_id": "2005.14165"}]},
                "selected_papers": {1: [{"arxiv_id": "2005.14165"}]},
                "iteration_metrics": {"discarded_gt_count": 0},
            },
        ])
        evaluator.deep_research_workflow = MagicMock()
        evaluator.deep_research_workflow.run.return_value = mock_result

        query_data = sample_benchmark_data[2]  # transformer_nlp: 3 GT papers
        result = evaluator.evaluate_single_query_deep_research(
            query_data, idx=2
        )

        assert result is not None
        assert len(result["iteration_results"]) == 3
        # Recall should improve: 1/3 -> 2/3 -> 3/3
        assert result["iteration_results"][0]["recall"] == pytest.approx(1 / 3, abs=1e-3)
        assert result["iteration_results"][1]["recall"] == pytest.approx(2 / 3, abs=1e-3)
        assert result["iteration_results"][2]["recall"] == pytest.approx(1.0)


# =============================================================================
# 9. CitationEvaluator - evaluate_benchmark aggregation (eval.py)
# =============================================================================
class TestEvaluateBenchmark:
    """Tests for CitationEvaluator.evaluate_benchmark aggregation logic."""

    def _make_evaluator(self):
        """Create a CitationEvaluator with mocked workflows."""
        import eval as eval_module
        mock_rag = MagicMock()
        mock_rag.get_available_search_methods.return_value = ["bm25"]

        eval_module.DeepResearchWorkflow = MagicMock
        eval_module.SimpleWorkflow = MagicMock

        from eval import CitationEvaluator
        evaluator = CitationEvaluator(
            rag_system=mock_rag
        )
        return evaluator

    def test_deep_research_avg_metrics(self, sample_benchmark_data):
        evaluator = self._make_evaluator()

        evaluator.deep_research_workflow = MagicMock()
        evaluator.deep_research_workflow.run.return_value = {
            "history": [
                {
                    "stage": "select",
                    "iter_idx": 1,
                    "retrieved_papers": {1: [{"arxiv_id": "2009.02040"}]},
                    "selected_papers": {1: [{"arxiv_id": "2009.02040"}]},
                    "iteration_metrics": {"discarded_gt_count": 0, "discarded_ratio": 0.0, "discarded_total_count": 0},
                    "subquery_metrics": {},
                    "gt_rank": [],
                    "browsing_arxiv_ids": [],
                    "avg_distance": 0.8,
                    "planner_during": 1.0,
                    "retrieval_during": 2.0,
                    "selector_during": 3.0,
                    "browser_during": -1,
                    "overhead_during": 0.5,
                    "total_during": 6.5,
                },
                {
                    "stage": "select",
                    "iter_idx": 2,
                    "retrieved_papers": {1: [{"arxiv_id": "1901.00137"}]},
                    "selected_papers": {1: [{"arxiv_id": "1901.00137"}]},
                    "iteration_metrics": {"discarded_gt_count": 0, "discarded_ratio": 0.0, "discarded_total_count": 0},
                    "subquery_metrics": {},
                    "gt_rank": [],
                    "browsing_arxiv_ids": [],
                    "avg_distance": 0.9,
                    "planner_during": 1.0,
                    "retrieval_during": 2.0,
                    "selector_during": 3.0,
                    "browser_during": -1,
                    "overhead_during": 0.5,
                    "total_during": 6.5,
                },
            ],
            "selected_papers": [Paper(id="p1", title="P1", abstract="", arxiv_id="2009.02040")],
            "executed_queries": [],
        }

        config.EVAL_MAX_ITERATIONS = 2
        results = evaluator.evaluate_benchmark(
            benchmark_data=sample_benchmark_data[:1],
            workflow="deep_research",
            enable_resume=False,
        )

        assert results["successful_queries"] == 1
        assert "avg_recall_iter_1" in results
        assert "avg_precision_iter_1" in results
        assert "avg_avg_distance_iter_1" in results
        assert results["avg_avg_distance_iter_1"] == pytest.approx(0.8)
        assert "avg_planner_during" in results
        assert results["avg_planner_during"] == pytest.approx(2.0)  # sum of 2 iters (1.0+1.0) / 1 query

    def test_simple_workflow_metrics(self, sample_benchmark_data):
        evaluator = self._make_evaluator()

        evaluator.simple_workflow = MagicMock()
        evaluator.simple_workflow.run.return_value = {
            "ground_truth_arxiv_ids": ["2009.02040", "1901.00137"],
            "top_results": [
                {"arxiv_id": "2009.02040", "title": "A", "similarity": 0.9, "matched": True},
                {"arxiv_id": "noise", "title": "N", "similarity": 0.5, "matched": False},
                {"arxiv_id": "1901.00137", "title": "B", "similarity": 0.4, "matched": True},
            ],
        }

        results = evaluator.evaluate_benchmark(
            benchmark_data=sample_benchmark_data[:1],
            workflow="simple",
            top_k_list=[2, 3],
            enable_resume=False,
        )

        assert results["successful_queries"] == 1
        assert results["avg_recall@2"] == pytest.approx(0.5)
        assert results["avg_precision@2"] == pytest.approx(0.5)
        assert results["avg_recall@3"] == pytest.approx(1.0)
        assert results["avg_precision@3"] == pytest.approx(2 / 3)

    def test_iteration_filling_for_early_stop(self, sample_benchmark_data):
        """When a query stops at iter 1 but max_iterations=3, iter 2 and 3 should inherit iter 1 values."""
        evaluator = self._make_evaluator()

        evaluator.deep_research_workflow = MagicMock()
        evaluator.deep_research_workflow.run.return_value = {
            "history": [
                {
                    "stage": "select",
                    "iter_idx": 1,
                    "retrieved_papers": {1: [{"arxiv_id": "2009.02040"}]},
                    "selected_papers": {1: [{"arxiv_id": "2009.02040"}]},
                    "iteration_metrics": {"discarded_gt_count": 0, "discarded_ratio": 0.1, "discarded_total_count": 3},
                    "subquery_metrics": {},
                    "gt_rank": [],
                    "browsing_arxiv_ids": [],
                    "avg_distance": 0.7,
                    "planner_during": 1.0,
                    "retrieval_during": 2.0,
                    "selector_during": 3.0,
                    "browser_during": -1,
                    "overhead_during": 0.5,
                    "total_during": 6.5,
                },
            ],
            "selected_papers": [Paper(id="p1", title="P1", abstract="", arxiv_id="2009.02040")],
            "executed_queries": [],
        }

        config.EVAL_MAX_ITERATIONS = 3
        results = evaluator.evaluate_benchmark(
            benchmark_data=sample_benchmark_data[:1],
            workflow="deep_research",
            enable_resume=False,
        )

        assert "avg_recall_iter_2" in results
        assert "avg_recall_iter_3" in results
        assert results["avg_recall_iter_1"] == pytest.approx(results["avg_recall_iter_2"])
        assert results["avg_recall_iter_1"] == pytest.approx(results["avg_recall_iter_3"])

    def test_failed_queries_count_as_zero(self, sample_benchmark_data):
        """Failed queries should dilute averages (divide by total, not successful)."""
        evaluator = self._make_evaluator()

        # First query succeeds with recall=1.0, second query fails (workflow returns None)
        evaluator.deep_research_workflow = MagicMock()
        evaluator.deep_research_workflow.run.side_effect = [
            {
                "history": [
                    {
                        "stage": "select",
                        "iter_idx": 1,
                        "retrieved_papers": {1: [{"arxiv_id": "2009.02040"}]},
                        "selected_papers": {1: [{"arxiv_id": "2009.02040"}]},
                        "iteration_metrics": {"discarded_gt_count": 0, "discarded_ratio": 0.0, "discarded_total_count": 0},
                        "subquery_metrics": {},
                        "gt_rank": [],
                        "browsing_arxiv_ids": [],
                        "avg_distance": 0.8,
                        "planner_during": 1.0,
                        "retrieval_during": 2.0,
                        "selector_during": 3.0,
                        "browser_during": -1,
                        "overhead_during": 0.5,
                        "total_during": 6.5,
                    },
                ],
                "selected_papers": [Paper(id="p1", title="P1", abstract="", arxiv_id="2009.02040")],
                "executed_queries": [],
            },
            None,  # Second query fails
        ]

        config.EVAL_MAX_ITERATIONS = 1
        results = evaluator.evaluate_benchmark(
            benchmark_data=sample_benchmark_data[:2],
            workflow="deep_research",
            enable_resume=False,
        )

        assert results["total_queries"] == 2
        assert results["successful_queries"] == 1
        # recall_iter_1 list has 1 entry (only the successful query)
        # but avg should divide by total_queries=2, not len=1
        assert results["avg_recall_iter_1"] == pytest.approx(
            results["recall_iter_1"][0] / 2
        )
