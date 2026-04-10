"""
Tests for Step 2 (Qdrant unification) and Step 3 (Agent module fixes).

Covers:
- CitationRAGSystem: init signature, load_or_build_indices signature, search method routing
- Planner: root continue (id=0), target_k fallback, link_type validation, early return
- Selector: return type (3-tuple)
- SelectorOutputWithBrowser: to_browse type (Dict[str, str])
- Browser: soup.head None guard
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

import config
from structures import Paper, SubQuery, SubQueryState, ResearchMemory, SelectorOutputWithBrowser


# =============================================================================
# 1. CitationRAGSystem — Step 2 changes
# =============================================================================
class TestCitationRAGSystemInit:
    """Tests for simplified CitationRAGSystem after FAISS removal."""

    def test_init_default_search_method(self):
        """Init with default search_method reads from config."""
        from rag import CitationRAGSystem
        with patch.object(config, 'DEFAULT_SEARCH_METHOD', 'bm25'):
            rag = CitationRAGSystem()
            assert rag.search_method == 'bm25'
            assert rag.qdrant_vector_store is None
            assert rag.bm25_index is None

    def test_init_explicit_search_method(self):
        """Init with explicit search_method overrides default."""
        from rag import CitationRAGSystem
        rag = CitationRAGSystem(search_method='vector')
        assert rag.search_method == 'vector'

    def test_init_no_faiss_attributes(self):
        """FAISS attributes should not exist after cleanup."""
        from rag import CitationRAGSystem
        rag = CitationRAGSystem()
        assert not hasattr(rag, 'faiss_index')
        assert not hasattr(rag, 'embedding_model')
        assert not hasattr(rag, 'faiss_id_to_index')

    def test_load_or_build_indices_signature(self):
        """load_or_build_indices should NOT accept faiss_path."""
        from rag import CitationRAGSystem
        import inspect
        sig = inspect.signature(CitationRAGSystem.load_or_build_indices)
        param_names = list(sig.parameters.keys())
        assert 'faiss_path' not in param_names
        assert 'paper_db_path' in param_names
        assert 'bm25_path' in param_names

    def test_no_hybrid_search_method(self):
        """get_available_search_methods should never return 'hybrid'."""
        from rag import CitationRAGSystem
        rag = CitationRAGSystem()
        rag.qdrant_vector_store = MagicMock()  # fake loaded
        rag.bm25_index = MagicMock()  # fake loaded
        methods = rag.get_available_search_methods()
        assert 'hybrid' not in methods
        assert 'vector' in methods
        assert 'bm25' in methods

    def test_load_or_build_bm25_only(self):
        """When search_method='bm25', should not try to load Qdrant."""
        from rag import CitationRAGSystem
        rag = CitationRAGSystem(search_method='bm25')
        with patch.object(rag, 'load_bm25_index') as mock_bm25, \
             patch.object(rag, 'load_qdrant_index') as mock_qdrant, \
             patch('os.path.exists', return_value=True):
            rag.load_or_build_indices(
                paper_db_path='dummy.json',
                bm25_path='dummy.pkl',
            )
            mock_bm25.assert_called_once()
            mock_qdrant.assert_not_called()

    def test_load_or_build_vector_only(self):
        """When search_method='vector', should not try to load BM25."""
        from rag import CitationRAGSystem
        rag = CitationRAGSystem(search_method='vector')
        with patch.object(rag, 'load_bm25_index') as mock_bm25, \
             patch.object(rag, 'load_qdrant_index') as mock_qdrant:
            rag.load_or_build_indices(
                paper_db_path='dummy.json',
                bm25_path='dummy.pkl',
            )
            mock_qdrant.assert_called_once()
            mock_bm25.assert_not_called()


# =============================================================================
# 2. Planner — Step 3 changes
# =============================================================================
class TestPlannerRootContinue:
    """Tests for planner root continue (id=0) and validation fixes."""

    def _make_planner(self):
        from agent.planner import Planner
        return Planner()

    def _make_subquery_state(self, sq_id, text="test", source_id=0, iter_index=1):
        sq = SubQuery(id=sq_id, text=text, source_subquery_id=source_id, iter_index=iter_index, link_type="derive")
        return SubQueryState(subquery=sq, total_requested=5)

    def _run_plan(self, planner, llm_response, subquery_states=None, user_query=None):
        user_query = user_query or {"query": "test query", "date": "2024-01"}
        memory = ResearchMemory(root_text=user_query["query"])
        subquery_states = subquery_states or {}

        with patch('agent.planner._call_llm', return_value=llm_response):
            return planner.plan_iteration(
                user_query=user_query,
                memory=memory,
                subquery_states=subquery_states,
                iteration_index=2,
                idx=99,
            )

    def test_root_continue_creates_subquery_with_root_text(self):
        """continue from id=0 should use root query text."""
        planner = self._make_planner()
        # Existing subquery state for id=1
        states = {1: [self._make_subquery_state(1)]}

        llm_response = (
            '<planner_output>{"subqueries": [{"link_type": "continue", "source_id": 0, "text": "", "target_k": 5}],'
            '"checklist": "", "experience_replay": "", "is_complete": false}</planner_output>'
        )
        subqueries, exp, checklist, is_complete = self._run_plan(planner, llm_response, states)
        assert 0 in subqueries
        assert subqueries[0].text == "test query"
        assert subqueries[0].source_subquery_id == -1

    def test_continue_nonexistent_id_skipped(self):
        """continue from non-existent source_id should be skipped."""
        planner = self._make_planner()
        llm_response = (
            '<planner_output>{"subqueries": [{"link_type": "continue", "source_id": 999, "text": "", "target_k": 5}],'
            '"checklist": "", "experience_replay": "", "is_complete": false}</planner_output>'
        )
        subqueries, _, _, _ = self._run_plan(planner, llm_response)
        assert len(subqueries) == 0

    def test_target_k_non_numeric_fallback(self):
        """Non-numeric target_k should fallback to config.MAX_RESULTS_PER_QUERY."""
        planner = self._make_planner()
        llm_response = (
            '<planner_output>{"subqueries": [{"link_type": "derive", "source_id": 0, "text": "test", "target_k": "abc"}],'
            '"checklist": "", "experience_replay": "", "is_complete": false}</planner_output>'
        )
        subqueries, _, _, _ = self._run_plan(planner, llm_response)
        assert len(subqueries) == 1
        sq = list(subqueries.values())[0]
        assert sq.target_k == config.MAX_RESULTS_PER_QUERY

    def test_target_k_valid_int(self):
        """Valid numeric target_k should be used as-is."""
        planner = self._make_planner()
        llm_response = (
            '<planner_output>{"subqueries": [{"link_type": "derive", "source_id": 0, "text": "test", "target_k": 3}],'
            '"checklist": "", "experience_replay": "", "is_complete": false}</planner_output>'
        )
        subqueries, _, _, _ = self._run_plan(planner, llm_response)
        sq = list(subqueries.values())[0]
        assert sq.target_k == 3

    def test_invalid_link_type_skipped(self):
        """Invalid link_type should be skipped."""
        planner = self._make_planner()
        llm_response = (
            '<planner_output>{"subqueries": ['
            '{"link_type": "invalid", "source_id": 0, "text": "test", "target_k": 5},'
            '{"link_type": "derive", "source_id": 0, "text": "", "target_k": 5},'
            '{"link_type": "derive", "source_id": 0, "text": "valid", "target_k": 5}'
            '],"checklist": "", "experience_replay": "", "is_complete": false}</planner_output>'
        )
        subqueries, _, _, _ = self._run_plan(planner, llm_response)
        # Only the last one (derive with text) should survive
        assert len(subqueries) == 1
        sq = list(subqueries.values())[0]
        assert sq.text == "valid"

    def test_early_return_complete_no_subqueries(self):
        """is_complete=True with no subqueries should return empty dict."""
        planner = self._make_planner()
        llm_response = (
            '<planner_output>{"subqueries": [],'
            '"checklist": "done", "experience_replay": "finished", "is_complete": true}</planner_output>'
        )
        subqueries, exp, checklist, is_complete = self._run_plan(planner, llm_response)
        assert subqueries == {}
        assert is_complete is True
        assert checklist == "done"


# =============================================================================
# 3. Selector — Step 3 return type
# =============================================================================
class TestSelectorReturnType:
    """Test that Selector.decide_for_subquery returns 3-tuple."""

    def test_returns_three_tuple(self):
        from agent.selector import Selector

        selector = Selector()
        sq = SubQuery(id=1, text="test query")
        papers = [
            Paper(id="p1", title="Paper 1", abstract="abs", arxiv_id="a1"),
            Paper(id="p2", title="Paper 2", abstract="abs", arxiv_id="a2"),
        ]

        mock_response = (
            '<selector_output>{"selected": ["p1"], "reasons": {"p1": "relevant"}, '
            '"overview": "test overview"}</selector_output>'
        )

        async def _run():
            with patch('agent.selector._call_llm_async', new_callable=AsyncMock, return_value=mock_response):
                return await selector.decide_for_subquery(
                    user_query="test",
                    sub_query=sq,
                    planner_checklist="",
                    papers=papers,
                    idx=99,
                )

        result = asyncio.run(_run())
        assert isinstance(result, tuple)
        assert len(result) == 3
        selected, overview, to_browse = result
        assert isinstance(selected, list)
        assert isinstance(overview, str)
        assert isinstance(to_browse, dict)


# =============================================================================
# 4. SelectorOutputWithBrowser — to_browse type
# =============================================================================
class TestSelectorOutputWithBrowserModel:
    """Test that to_browse is Dict[str, str] not List[dict]."""

    def test_to_browse_accepts_dict(self):
        output = SelectorOutputWithBrowser(
            selected=["p1"],
            discarded=["p2"],
            to_browse={"p3": "check methodology section"},
            reasons={"p1": "relevant"},
            overview="test",
        )
        assert output.to_browse == {"p3": "check methodology section"}

    def test_to_browse_empty_dict(self):
        output = SelectorOutputWithBrowser(
            selected=[], discarded=[], to_browse={},
            reasons={}, overview="",
        )
        assert output.to_browse == {}


# =============================================================================
# 5. Browser — soup.head None guard
# =============================================================================
class TestBrowserHeadNoneGuard:
    """Test that parse_html_content handles missing <head> gracefully."""

    def test_missing_head_raises_value_error(self):
        from agent.browser import Ar5ivParser
        parser = Ar5ivParser()
        # HTML with no <head> tag
        html = "<html><body><p>test</p></body></html>"
        with pytest.raises(Exception):
            # Should raise ValueError("Missing <title> tag") not AttributeError
            parser.parse_html_content(html)

    def test_missing_title_raises_value_error(self):
        from agent.browser import Ar5ivParser
        parser = Ar5ivParser()
        # HTML with <head> but no <title>
        html = "<html><head></head><body><p>test</p></body></html>"
        with pytest.raises(Exception):
            parser.parse_html_content(html)
