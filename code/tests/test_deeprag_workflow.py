"""
Tests for DeepResearchWorkflow (code/deeprag.py).

Covers:
- _mcp_results_to_papers: dict -> Paper conversion
- _process_selector_results: dedup/merge, subquery state update, browsing task collection
- run() orchestration: multi-iteration loop with mocked planner/selector/retrieval

Test data is loaded from code/tests/fixtures/deeprag_cases.json via conftest.py fixtures.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
from conftest import load_fixture

# conftest.py already sets sys.path and injects a minimal config module
import config
from structures import Paper, SubQuery, SubQueryState, ResearchMemory
from deeprag import DeepResearchWorkflow


# ---------------------------------------------------------------------------
# Load parametrize data from fixture files
# ---------------------------------------------------------------------------
_deeprag_data = load_fixture("deeprag_cases.json")


def _param_id(case):
    return case["id"]


def _make_paper(id="p1", title="Paper 1", abstract="abs", arxiv_id="a1", **kwargs):
    """Helper to create a Paper object."""
    return Paper(id=id, title=title, abstract=abstract, arxiv_id=arxiv_id, **kwargs)


def _paper_from_dict(d):
    """Create a Paper from a fixture dict."""
    return Paper(
        id=d.get("id", ""),
        title=d.get("title", "N/A"),
        abstract=d.get("abstract", "abs"),
        arxiv_id=d.get("arxiv_id", "N/A"),
        date=d.get("date"),
        score=d.get("score"),
    )


def _papers_registry():
    """Build a paper registry from the fixture's process_selector_results.papers."""
    registry = {}
    for key, pdata in _deeprag_data["process_selector_results"]["papers"].items():
        registry[key] = _paper_from_dict(pdata)
    return registry


# =============================================================================
# 1. _mcp_results_to_papers
# =============================================================================
class TestMcpResultsToPapers:
    """Tests for DeepResearchWorkflow._mcp_results_to_papers."""

    def _make_workflow(self):
        mock_rag = MagicMock()
        return DeepResearchWorkflow(
            rag_system=mock_rag, llm_model="test", gen_params={}, is_local=True
        )

    @pytest.mark.parametrize("case", _deeprag_data["mcp_results_to_papers"], ids=_param_id)
    def test_conversion(self, case):
        wf = self._make_workflow()
        papers = wf._mcp_results_to_papers(case["input"])
        assert len(papers) == len(case["expected"])

        for paper, exp in zip(papers, case["expected"]):
            if "id" in exp:
                assert paper.id == exp["id"]
            if "title" in exp:
                assert paper.title == exp["title"]
            if "abstract" in exp:
                assert paper.abstract == exp["abstract"]
            if "arxiv_id" in exp:
                assert paper.arxiv_id == exp["arxiv_id"]
            if "date" in exp:
                assert paper.date == exp["date"]
            if "score" in exp:
                assert paper.score == exp["score"]


# =============================================================================
# 2. _process_selector_results
# =============================================================================
class TestProcessSelectorResults:
    """Tests for DeepResearchWorkflow._process_selector_results."""

    def _build_selection_results(self, case_data, papers_reg):
        """Convert fixture case data into the selection_results dict expected by the method."""
        selection_results = {}
        for sq_id_str, val in case_data["selection_results"].items():
            sq_id = int(sq_id_str)
            if val is None:
                selection_results[sq_id] = None
            else:
                kept = [papers_reg[pid] for pid in val["kept_ids"]]
                overview = val["overview"]
                to_browse = {}
                for bid, bdata in val.get("to_browse", {}).items():
                    to_browse[bid] = {"paper": papers_reg[bdata["paper_id"]], "goal": bdata["goal"]} if isinstance(bdata, dict) else bdata
                selection_results[sq_id] = (kept, overview, to_browse)
        return selection_results

    def _build_existing_selected(self, case_data, papers_reg):
        cur = {}
        for sq_id_str, paper_ids in case_data.get("existing_selected", {}).items():
            cur[int(sq_id_str)] = [papers_reg[pid] for pid in paper_ids]
        return cur

    def _build_subquery_states(self, case_data):
        states = {}
        for sq_id_str, state_list in case_data.get("subquery_states", {}).items():
            sq_id = int(sq_id_str)
            states[sq_id] = [
                SubQueryState(subquery=SubQuery(id=sq_id, text=s["text"]))
                for s in state_list
            ]
        return states

    @pytest.mark.parametrize(
        "case",
        _deeprag_data["process_selector_results"]["cases"],
        ids=_param_id,
    )
    def test_process_selector(self, case):
        papers_reg = _papers_registry()
        selection_results = self._build_selection_results(case, papers_reg)
        cur_selected = self._build_existing_selected(case, papers_reg)
        sq_states = self._build_subquery_states(case)
        browsing = {}

        DeepResearchWorkflow._process_selector_results(
            selection_results=selection_results,
            cur_iter_selected_papers=cur_selected,
            subquery_states=sq_states,
            checklist="checklist",
            papers_for_browsing=browsing,
        )

        for sq_id_str, expected_count in case["expected_count"].items():
            sq_id = int(sq_id_str)
            assert len(cur_selected.get(sq_id, [])) == expected_count, \
                f"sq_id={sq_id}: expected {expected_count} papers, got {len(cur_selected.get(sq_id, []))}"

        if "expected_first_id" in case:
            for sq_id_str, first_id in case["expected_first_id"].items():
                sq_id = int(sq_id_str)
                assert cur_selected[sq_id][0].id == first_id

        if "expected_overviews" in case:
            for sq_id_str, overview in case["expected_overviews"].items():
                sq_id = int(sq_id_str)
                if sq_id in sq_states and sq_states[sq_id]:
                    assert sq_states[sq_id][-1].selector_overview == overview

    @pytest.mark.parametrize(
        "case", _deeprag_data["incremental_mode_cases"], ids=_param_id
    )
    def test_incremental_mode(self, case):
        """INCREMENTAL mode: merge with deduplication."""
        config.BROWSER_MODE = "INCREMENTAL"
        try:
            papers_reg = _papers_registry()
            cur_selected = {}
            for sq_id_str, paper_ids in case.get("existing", {}).items():
                cur_selected[int(sq_id_str)] = [papers_reg[pid] for pid in paper_ids]

            kept = [papers_reg[pid] for pid in case["new_kept_ids"]]
            sq_states = {1: [SubQueryState(subquery=SubQuery(id=1, text="q1"))]}
            browsing = {}

            DeepResearchWorkflow._process_selector_results(
                selection_results={1: (kept, "ov", {})},
                cur_iter_selected_papers=cur_selected,
                subquery_states=sq_states,
                checklist="c",
                papers_for_browsing=browsing,
            )

            assert len(cur_selected[1]) == case["expected_count"]
            actual_ids = [p.id for p in cur_selected[1]]
            assert actual_ids == case["expected_ids"]
        finally:
            config.BROWSER_MODE = "NONE"

    def test_browsing_tasks_collected(self):
        """to_browse dict should be transferred to papers_for_browsing."""
        papers_reg = _papers_registry()
        p1 = papers_reg["p1"]
        browse_item = {"paper": p1, "goal": "check methodology"}

        cur_selected = {}
        sq_states = {1: [SubQueryState(subquery=SubQuery(id=1, text="q1"))]}
        browsing = {}

        DeepResearchWorkflow._process_selector_results(
            selection_results={1: ([], "ov", {"p1": browse_item})},
            cur_iter_selected_papers=cur_selected,
            subquery_states=sq_states,
            checklist="c",
            papers_for_browsing=browsing,
        )

        assert 1 in browsing
        assert len(browsing[1]) == 1
        assert browsing[1][0]["goal"] == "check methodology"

    def test_no_browsing_tasks_when_empty(self):
        """Empty to_browse should not modify papers_for_browsing."""
        cur_selected = {}
        sq_states = {1: [SubQueryState(subquery=SubQuery(id=1, text="q1"))]}
        browsing = {}

        DeepResearchWorkflow._process_selector_results(
            selection_results={1: ([_make_paper()], "ov", {})},
            cur_iter_selected_papers=cur_selected,
            subquery_states=sq_states,
            checklist="c",
            papers_for_browsing=browsing,
        )

        assert browsing == {}


# =============================================================================
# 3. run() orchestration (mocked agents)
# =============================================================================
class TestDeepResearchWorkflowRun:
    """Tests for DeepResearchWorkflow.run() with fully mocked agents."""

    def _make_workflow(self):
        mock_rag = MagicMock()
        wf = DeepResearchWorkflow(
            rag_system=mock_rag, llm_model="test", gen_params={}, is_local=True
        )
        return wf

    def _paper_from_fixture(self, key):
        """Get a Paper from the workflow_run_papers fixture data."""
        d = _deeprag_data["workflow_run_papers"][key]
        return _paper_from_dict(d)

    def _subquery_from_fixture(self, key):
        """Get a SubQuery from the workflow_run_subqueries fixture data."""
        d = _deeprag_data["workflow_run_subqueries"][key]
        if isinstance(d, dict) and "id" in d:
            return SubQuery(
                id=d["id"], text=d["text"], target_k=d.get("target_k", 5),
                iter_index=d.get("iter_index"),
            )
        # It's a dict of subqueries
        result = {}
        for sq_key, sq_data in d.items():
            sq_id = int(sq_key) if sq_key.isdigit() else sq_data["id"]
            result[sq_id] = SubQuery(
                id=sq_data["id"], text=sq_data["text"], target_k=sq_data.get("target_k", 5),
                iter_index=sq_data.get("iter_index"),
            )
        return result

    def _setup_one_iteration_mocks(self, wf, selected_papers=None, subqueries=None):
        if subqueries is None:
            subqueries = self._subquery_from_fixture("default")
        if selected_papers is None:
            p = self._paper_from_fixture("default_paper")
            selected_papers = {1: [p]}

        wf.planner = MagicMock()
        wf.planner.plan_iteration.return_value = (
            subqueries, "experience replay text", "checklist items", False,
        )

        async def mock_select(*args, **kwargs):
            sub_query = kwargs.get("sub_query") or (args[2] if len(args) > 2 else None)
            sq_id = sub_query.id if sub_query else 1
            kept = selected_papers.get(sq_id, [])
            return (kept, "overview", {})

        wf.selector = MagicMock()
        wf.selector.decide_for_subquery = AsyncMock(side_effect=mock_select)

        return subqueries, selected_papers

    @patch("deeprag.search_papers")
    def test_single_iteration_returns_expected_structure(self, mock_search):
        wf = self._make_workflow()
        self._setup_one_iteration_mocks(wf)
        mock_search.return_value = ([], {})

        result = wf.run(
            query={"query": "test query"},
            gt_arxiv_ids={"2001.00001"},
            max_iterations=1, idx=0,
        )

        assert result is not None
        assert "history" in result
        assert "selected_papers" in result
        assert "executed_queries" in result
        assert len(result["history"]) == 2  # 1 plan + 1 select

    @patch("deeprag.search_papers")
    def test_planner_complete_stops_early(self, mock_search):
        wf = self._make_workflow()
        wf.planner = MagicMock()
        wf.planner.plan_iteration.return_value = ({}, "experience", "checklist", True)
        mock_search.return_value = ([], {})

        result = wf.run(query={"query": "test"}, gt_arxiv_ids={"a1"}, max_iterations=3, idx=0)
        assert result is None

    @patch("deeprag.search_papers")
    def test_planner_complete_with_papers_returns_early(self, mock_search):
        wf = self._make_workflow()

        sq_data = _deeprag_data["workflow_run_subqueries"]["default"]
        subqueries = {}
        for k, v in sq_data.items():
            subqueries[int(k)] = SubQuery(id=v["id"], text=v["text"], target_k=v.get("target_k", 5))

        call_count = [0]
        def plan_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (subqueries, "exp", "checklist", False)
            else:
                return ({}, "exp", "checklist", True)

        wf.planner = MagicMock()
        wf.planner.plan_iteration.side_effect = plan_side_effect

        wf.selector = MagicMock()
        wf.selector.decide_for_subquery = AsyncMock(
            return_value=([self._paper_from_fixture("default_paper")], "ov", {})
        )
        mock_search.return_value = ([], {})

        result = wf.run(query={"query": "test"}, gt_arxiv_ids={"a1"}, max_iterations=5, idx=0)

        assert result is not None
        plan_stages = [h for h in result["history"] if h["stage"] == "plan"]
        assert len(plan_stages) == 2
        assert wf.planner.plan_iteration.call_count == 2

    @patch("deeprag.search_papers")
    def test_no_subqueries_returns_none(self, mock_search):
        wf = self._make_workflow()
        wf.planner = MagicMock()
        wf.planner.plan_iteration.return_value = ({}, "exp", "checklist", False)

        result = wf.run(query={"query": "test"}, max_iterations=1, idx=0)
        assert result is None

    @patch("deeprag.search_papers")
    def test_multi_iteration_paper_accumulation(self, mock_search):
        wf = self._make_workflow()

        p1 = self._paper_from_fixture("paper_a1")
        p2 = self._paper_from_fixture("paper_a2")

        sq_data = _deeprag_data["workflow_run_subqueries"]["default"]
        subqueries = {}
        for k, v in sq_data.items():
            subqueries[int(k)] = SubQuery(id=v["id"], text=v["text"], target_k=v.get("target_k", 5))

        wf.planner = MagicMock()
        wf.planner.plan_iteration.return_value = (subqueries, "exp", "checklist", False)

        call_iter = [0]
        async def mock_select(*args, **kwargs):
            call_iter[0] += 1
            if call_iter[0] == 1:
                return ([p1], "ov1", {})
            else:
                return ([p2], "ov2", {})

        wf.selector = MagicMock()
        wf.selector.decide_for_subquery = AsyncMock(side_effect=mock_select)
        mock_search.return_value = ([], {})

        result = wf.run(
            query={"query": "test"}, gt_arxiv_ids={"a1", "a2"}, max_iterations=2, idx=0,
        )

        selected_ids = {p.id for p in result["selected_papers"]}
        assert "p1" in selected_ids
        assert "p2" in selected_ids

    @patch("deeprag.search_papers")
    def test_gt_metrics_in_history(self, mock_search):
        wf = self._make_workflow()
        self._setup_one_iteration_mocks(wf)
        mock_search.return_value = ([], {})

        result = wf.run(
            query={"query": "test"}, gt_arxiv_ids={"a1"}, max_iterations=1, idx=0,
        )

        select_stages = [h for h in result["history"] if h["stage"] == "select"]
        assert len(select_stages) == 1
        s = select_stages[0]
        for field in ["iteration_metrics", "avg_distance", "gt_rank", "retrieved_papers", "selected_papers"]:
            assert field in s

    @patch("deeprag.search_papers")
    def test_timing_fields_in_history(self, mock_search):
        wf = self._make_workflow()
        self._setup_one_iteration_mocks(wf)
        mock_search.return_value = ([], {})

        result = wf.run(
            query={"query": "test"}, gt_arxiv_ids=set(), max_iterations=1, idx=0,
        )

        select_stages = [h for h in result["history"] if h["stage"] == "select"]
        s = select_stages[0]
        for field in ["planner_during", "retrieval_during", "selector_during",
                       "browser_during", "overhead_during", "total_during"]:
            assert field in s
            assert isinstance(s[field], float)

    @patch("deeprag.search_papers")
    def test_memory_updated_across_iterations(self, mock_search):
        wf = self._make_workflow()

        iter_sq = _deeprag_data["workflow_run_subqueries"]["iter_queries"]
        sq1 = SubQuery(id=iter_sq["sq1"]["id"], text=iter_sq["sq1"]["text"],
                       target_k=iter_sq["sq1"]["target_k"], iter_index=iter_sq["sq1"]["iter_index"])
        sq2 = SubQuery(id=iter_sq["sq2"]["id"], text=iter_sq["sq2"]["text"],
                       target_k=iter_sq["sq2"]["target_k"], iter_index=iter_sq["sq2"]["iter_index"])

        call_count = [0]
        def plan_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ({1: sq1}, "exp1", "check1", False)
            else:
                return ({2: sq2}, "exp2", "check2", False)

        wf.planner = MagicMock()
        wf.planner.plan_iteration.side_effect = plan_side_effect
        wf.selector = MagicMock()
        wf.selector.decide_for_subquery = AsyncMock(
            return_value=([self._paper_from_fixture("default_paper")], "ov", {})
        )
        mock_search.return_value = ([], {})

        result = wf.run(
            query={"query": "test"}, gt_arxiv_ids=set(), max_iterations=2, idx=0,
        )

        plan_stages = [h for h in result["history"] if h["stage"] == "plan"]
        assert len(plan_stages) == 2
        assert any(sq["id"] == 1 for sq in plan_stages[0]["sub_queries"])
        assert any(sq["id"] == 2 for sq in plan_stages[1]["sub_queries"])

    @patch("deeprag.search_papers")
    def test_executed_queries_tracked(self, mock_search):
        wf = self._make_workflow()

        two_q = _deeprag_data["workflow_run_subqueries"]["two_queries"]
        subqueries = {}
        for k, v in two_q.items():
            subqueries[int(k)] = SubQuery(id=v["id"], text=v["text"], target_k=v.get("target_k", 5))

        wf.planner = MagicMock()
        wf.planner.plan_iteration.return_value = (subqueries, "exp", "checklist", False)
        wf.selector = MagicMock()
        wf.selector.decide_for_subquery = AsyncMock(
            side_effect=[
                ([self._paper_from_fixture("default_paper")], "ov", {}),
                ([self._paper_from_fixture("paper_2")], "ov", {}),
            ]
        )
        mock_search.return_value = ([], {})

        result = wf.run(
            query={"query": "test"}, gt_arxiv_ids=set(), max_iterations=1, idx=0,
        )

        assert "neural network" in result["executed_queries"]
        assert "deep learning" in result["executed_queries"]

    @patch("deeprag.search_papers")
    def test_final_selected_papers_deduplicated(self, mock_search):
        wf = self._make_workflow()

        p1 = self._paper_from_fixture("paper_a1")

        sq_data = _deeprag_data["workflow_run_subqueries"]["default"]
        subqueries = {}
        for k, v in sq_data.items():
            subqueries[int(k)] = SubQuery(id=v["id"], text=v["text"], target_k=v.get("target_k", 5))

        wf.planner = MagicMock()
        wf.planner.plan_iteration.return_value = (subqueries, "exp", "checklist", False)

        wf.selector = MagicMock()
        wf.selector.decide_for_subquery = AsyncMock(return_value=([p1], "ov", {}))
        mock_search.return_value = ([], {})

        result = wf.run(
            query={"query": "test"}, gt_arxiv_ids=set(), max_iterations=2, idx=0,
        )

        selected_ids = [p.id for p in result["selected_papers"]]
        assert selected_ids.count("p1") == 1

    @patch("deeprag.search_papers")
    def test_single_iteration_with_empty_gt(self, mock_search):
        """run() with empty gt_arxiv_ids should still work."""
        wf = self._make_workflow()
        self._setup_one_iteration_mocks(wf)
        mock_search.return_value = ([], {})

        result = wf.run(
            query={"query": "test"}, gt_arxiv_ids=set(), max_iterations=1, idx=0,
        )

        assert result is not None
        select_stages = [h for h in result["history"] if h["stage"] == "select"]
        assert select_stages[0]["avg_distance"] == pytest.approx(0.0)

    @patch("deeprag.search_papers")
    def test_three_iterations_run(self, mock_search):
        """Verify 3 iterations produce 3 plan + 3 select stages."""
        wf = self._make_workflow()

        sq_data = _deeprag_data["workflow_run_subqueries"]["default"]
        subqueries = {}
        for k, v in sq_data.items():
            subqueries[int(k)] = SubQuery(id=v["id"], text=v["text"], target_k=v.get("target_k", 5))

        wf.planner = MagicMock()
        wf.planner.plan_iteration.return_value = (subqueries, "exp", "checklist", False)
        wf.selector = MagicMock()
        wf.selector.decide_for_subquery = AsyncMock(
            return_value=([self._paper_from_fixture("default_paper")], "ov", {})
        )
        mock_search.return_value = ([], {})

        result = wf.run(
            query={"query": "test"}, gt_arxiv_ids=set(), max_iterations=3, idx=0,
        )

        plan_stages = [h for h in result["history"] if h["stage"] == "plan"]
        select_stages = [h for h in result["history"] if h["stage"] == "select"]
        assert len(plan_stages) == 3
        assert len(select_stages) == 3


# =============================================================================
# 4. BROWSER_MODE orchestration tests
# =============================================================================
class TestBrowserModeOrchestration:
    """
    Tests that verify the different BROWSER_MODE data flows at the run() level:

    - NONE:          selector once, no browsing, overwrite
    - PRE_ENRICH:    browse ALL papers → selector once, overwrite
    - REFRESH:       selector → browse uncertain → re-selector with ALL original papers, overwrite
    - INCREMENTAL:   selector → browse uncertain → re-selector with ONLY browsed papers, merge+dedup
    """

    def _make_workflow(self):
        mock_rag = MagicMock()
        return DeepResearchWorkflow(
            rag_system=mock_rag, llm_model="test", gen_params={}, is_local=True
        )

    def _paper_from_fixture(self, key):
        d = _deeprag_data["workflow_run_papers"][key]
        return _paper_from_dict(d)

    def _build_papers_map(self, paper_keys):
        """Build a {paper_key: Paper} lookup from fixture keys."""
        return {k: self._paper_from_fixture(k) for k in paper_keys}

    @pytest.mark.parametrize(
        "case", _deeprag_data["browser_mode_cases"], ids=_param_id
    )
    @patch("deeprag.search_papers")
    def test_browser_mode(self, mock_search, case):
        mode = case["mode"]
        config.BROWSER_MODE = mode
        try:
            wf = self._make_workflow()

            # Build papers from fixture keys
            all_paper_keys = set(case["initial_papers"])
            if case.get("selector_second_kept"):
                all_paper_keys.update(case["selector_second_kept"])
            papers_map = self._build_papers_map(all_paper_keys)

            initial_papers = [papers_map[k] for k in case["initial_papers"]]

            # Planner returns one subquery
            sq = SubQuery(id=1, text="test query", target_k=5)
            wf.planner = MagicMock()
            wf.planner.plan_iteration.return_value = ({1: sq}, "exp", "checklist", False)

            # Mock search to return initial papers
            mock_search.return_value = (
                [{"paper_id": p.id, "title": p.title, "abstract": p.abstract,
                  "arxiv_id": p.arxiv_id} for p in initial_papers],
                {}
            )

            # Build selector mock with first/second call behavior
            first_kept = [papers_map[k] for k in case["selector_first_kept"]]
            first_to_browse = {}
            for paper_key, goal in case.get("selector_first_to_browse", {}).items():
                first_to_browse[paper_key] = {"paper": papers_map[paper_key], "goal": goal}

            selector_call_count = [0]
            selector_received_papers = []  # track what papers were passed to selector

            async def mock_selector(*args, **kwargs):
                selector_call_count[0] += 1
                # Record the papers argument passed to selector
                papers_arg = kwargs.get("papers") or (args[3] if len(args) > 3 else [])
                selector_received_papers.append(list(papers_arg))

                if selector_call_count[0] == 1:
                    return (first_kept, "overview1", first_to_browse)
                else:
                    second_kept = [papers_map[k] for k in case.get("selector_second_kept", [])]
                    second_to_browse = {}
                    for paper_key, goal in case.get("selector_second_to_browse", {}).items():
                        second_to_browse[paper_key] = {"paper": papers_map[paper_key], "goal": goal}
                    return (second_kept, "overview2", second_to_browse)

            wf.selector = MagicMock()
            wf.selector.decide_for_subquery = AsyncMock(side_effect=mock_selector)

            # Mock browser
            browse_call_count = [0]
            browse_received_papers = []

            async def mock_browse(*args, **kwargs):
                browse_call_count[0] += 1
                paper_arg = kwargs.get("paper") or (args[1] if len(args) > 1 else None)
                browse_received_papers.append(paper_arg)

            wf.browser = MagicMock()
            wf.browser.browse_papers = AsyncMock(side_effect=mock_browse)

            result = wf.run(
                query={"query": "test"}, gt_arxiv_ids=set(), max_iterations=1, idx=0,
            )

            assert result is not None

            # Verify browsing was/wasn't called
            if case["expect_browse_called"]:
                assert browse_call_count[0] > 0, f"Expected browsing in {mode} mode"
                if "expect_browsed_paper_count" in case:
                    assert browse_call_count[0] == case["expect_browsed_paper_count"], \
                        f"Expected {case['expect_browsed_paper_count']} browse calls, got {browse_call_count[0]}"
            else:
                assert browse_call_count[0] == 0, f"Did not expect browsing in {mode} mode"

            # Verify selector call count
            assert selector_call_count[0] == case["expect_selector_calls"], \
                f"Expected {case['expect_selector_calls']} selector calls, got {selector_call_count[0]}"

            # Verify second selector received correct paper count (REFRESH vs INCREMENTAL)
            if case.get("expect_second_selector_paper_count") is not None and len(selector_received_papers) >= 2:
                actual = len(selector_received_papers[1])
                expected = case["expect_second_selector_paper_count"]
                assert actual == expected, \
                    f"Second selector call: expected {expected} papers, got {actual}"

            # Verify final selected papers
            selected_ids = sorted([p.id for p in result["selected_papers"]])
            expected_ids = sorted(case["expected_selected_ids"])
            assert selected_ids == expected_ids, \
                f"Expected selected {expected_ids}, got {selected_ids}"

            # No duplicate papers in final result
            assert len(selected_ids) == len(set(selected_ids)), "Duplicate papers in final selection"

        finally:
            config.BROWSER_MODE = "NONE"
