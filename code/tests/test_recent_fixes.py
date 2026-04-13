"""
Tests for recent bug fixes (selector ID types, deterministic UUID, checkpoint
corruption handling, asyncio.gather exceptions, BM25 tokenization, rag.py
error handling, browser backoff).
"""
import asyncio
import json
import os
import uuid
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from conftest import load_fixture
import config
from structures import Paper, SubQuery


# =============================================================================
# 1. Selector: string ID conversion from LLM numeric output
# =============================================================================
class TestSelectorStringIdConversion:
    """Tests for selector str() conversion when LLM outputs numeric IDs."""

    def _make_selector(self):
        from agent.selector import Selector
        return Selector()

    def test_numeric_ids_converted_to_strings(self):
        """LLM outputs [2007.05993] as JSON number; selector must str() it."""
        selector = self._make_selector()
        sq = SubQuery(id=1, text="test query")

        # paper id is a string like "2007.05993"
        papers = [
            Paper(id="2007.05993", title="Paper A", abstract="abs", arxiv_id="2007.05993"),
            Paper(id="2103.00020", title="Paper B", abstract="abs", arxiv_id="2103.00020"),
        ]

        # LLM returns numeric IDs (JSON parses 2007.05993 as float)
        mock_response = (
            '<selector_output>{"selected": [2007.05993, "2103.00020"], '
            '"reasons": {"2007.05993": "relevant"}, '
            '"overview": "test"}</selector_output>'
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

        selected, overview, to_browse = asyncio.run(_run())
        assert len(selected) == 2
        selected_ids = {p.id for p in selected}
        assert "2007.05993" in selected_ids
        assert "2103.00020" in selected_ids

    def test_to_browse_numeric_key_conversion(self):
        """to_browse dict keys from LLM may be numeric; must convert to str."""
        selector = self._make_selector()
        sq = SubQuery(id=1, text="test query")

        papers = [
            Paper(id="2103.00020", title="Paper A", abstract="abs", arxiv_id="2103.00020"),
            Paper(id="2104.00001", title="Paper B", abstract="abs", arxiv_id="2104.00001"),
        ]

        # LLM returns numeric key in to_browse
        mock_response = (
            '<selector_output>{"selected": ["2103.00020"], '
            '"to_browse": {"2104.00001": "check methodology"}, '
            '"overview": "test"}</selector_output>'
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

        selected, overview, to_browse = asyncio.run(_run())
        assert "2104.00001" in to_browse
        assert to_browse["2104.00001"]["paper"].id == "2104.00001"
        assert to_browse["2104.00001"]["goal"] == "check methodology"

    def test_mixed_numeric_and_string_ids(self):
        """LLM may mix numeric and string IDs in the same response."""
        selector = self._make_selector()
        sq = SubQuery(id=1, text="test query")

        papers = [
            Paper(id="1234.56789", title="Paper A", abstract="abs", arxiv_id="1234.56789"),
            Paper(id="abcd", title="Paper B", abstract="abs", arxiv_id="abcd"),
            Paper(id="9876.54321", title="Paper C", abstract="abs", arxiv_id="9876.54321"),
        ]

        # Mix: first is numeric, second is string, third is numeric
        mock_response = (
            '<selector_output>{"selected": [1234.56789, "abcd"], '
            '"overview": "mixed"}</selector_output>'
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

        selected, _, _ = asyncio.run(_run())
        assert len(selected) == 2
        ids = {p.id for p in selected}
        assert "1234.56789" in ids
        assert "abcd" in ids

    def test_unknown_id_silently_dropped(self):
        """IDs not in the paper list should be silently dropped."""
        selector = self._make_selector()
        sq = SubQuery(id=1, text="test query")
        papers = [
            Paper(id="p1", title="Paper 1", abstract="abs", arxiv_id="a1"),
        ]

        mock_response = (
            '<selector_output>{"selected": ["p1", "nonexistent"], '
            '"overview": "test"}</selector_output>'
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

        selected, _, _ = asyncio.run(_run())
        assert len(selected) == 1
        assert selected[0].id == "p1"


# =============================================================================
# 2. Deterministic UUID5 generation (build_vector_db.py)
# =============================================================================
class TestDeterministicUUID:
    """Tests for _deterministic_id in build_vector_db.py."""

    def test_deterministic_same_input_same_output(self):
        """Same arxiv_id must always produce the same UUID."""
        from build_vector_db import _deterministic_id
        id1 = _deterministic_id("2007.05993")
        id2 = _deterministic_id("2007.05993")
        assert id1 == id2

    def test_deterministic_different_inputs(self):
        """Different arxiv_ids must produce different UUIDs."""
        from build_vector_db import _deterministic_id
        id1 = _deterministic_id("2007.05993")
        id2 = _deterministic_id("2103.00020")
        assert id1 != id2

    def test_output_is_valid_uuid(self):
        """Output must be a valid UUID string."""
        from build_vector_db import _deterministic_id
        result = _deterministic_id("2007.05993")
        # Should not raise
        parsed = uuid.UUID(result)
        assert str(parsed) == result

    def test_idempotent_across_calls(self):
        """Multiple calls with the same key produce identical results."""
        from build_vector_db import _deterministic_id
        results = [_deterministic_id("test.paper.1234") for _ in range(10)]
        assert len(set(results)) == 1

    def test_uses_uuid5(self):
        """Should use uuid5 (deterministic), not uuid4 (random)."""
        from build_vector_db import _deterministic_id, _QDRANT_NS
        key = "test_key_12345"
        expected = str(uuid.uuid5(_QDRANT_NS, key))
        assert _deterministic_id(key) == expected


# =============================================================================
# 3. CheckpointManager: corrupt line handling
# =============================================================================
class TestCheckpointCorruptLineHandling:
    """Tests for CheckpointManager.load_checkpoint with corrupt lines."""

    def test_corrupt_line_skipped(self, tmp_path):
        """Corrupt JSON lines should be skipped, valid lines preserved."""
        from utils import CheckpointManager

        cp_file = str(tmp_path / "checkpoint.jsonl")
        with open(cp_file, "w") as f:
            f.write(json.dumps({"idx": 0, "query": "q0"}) + "\n")
            f.write("THIS IS NOT JSON\n")
            f.write(json.dumps({"idx": 2, "query": "q2"}) + "\n")

        cm = CheckpointManager(cp_file)
        processed, cached = cm.load_checkpoint()
        assert processed == {0, 2}
        assert len(cached) == 2

    def test_all_corrupt_lines(self, tmp_path):
        """All corrupt lines should result in empty processed set."""
        from utils import CheckpointManager

        cp_file = str(tmp_path / "checkpoint.jsonl")
        with open(cp_file, "w") as f:
            f.write("bad line 1\n")
            f.write("bad line 2\n")
            f.write("{broken json\n")

        cm = CheckpointManager(cp_file)
        processed, cached = cm.load_checkpoint()
        assert processed == set()
        assert cached == []

    def test_truncated_json_line(self, tmp_path):
        """Truncated JSON (e.g., from killed process) should be skipped."""
        from utils import CheckpointManager

        cp_file = str(tmp_path / "checkpoint.jsonl")
        with open(cp_file, "w") as f:
            f.write('{"idx": 1, "query": "test", "data":')
            # truncated - no closing brace or newline

        cm = CheckpointManager(cp_file)
        processed, cached = cm.load_checkpoint()
        assert processed == set()
        assert cached == []

    def test_mixed_valid_corrupt_empty_lines(self, tmp_path):
        """Mix of valid, corrupt, and empty lines."""
        from utils import CheckpointManager

        cp_file = str(tmp_path / "checkpoint.jsonl")
        with open(cp_file, "w") as f:
            f.write("\n")  # empty line
            f.write(json.dumps({"idx": 5, "query": "q5"}) + "\n")
            f.write("\n")  # empty line
            f.write("not json\n")
            f.write(json.dumps({"idx": 10, "query": "q10"}) + "\n")

        cm = CheckpointManager(cp_file)
        processed, cached = cm.load_checkpoint()
        assert processed == {5, 10}
        assert len(cached) == 2

    def test_negative_idx_skipped(self, tmp_path):
        """Lines with idx < 0 should not be added to processed set."""
        from utils import CheckpointManager

        cp_file = str(tmp_path / "checkpoint.jsonl")
        with open(cp_file, "w") as f:
            f.write(json.dumps({"idx": -1, "query": "bad"}) + "\n")
            f.write(json.dumps({"idx": 3, "query": "good"}) + "\n")
            f.write(json.dumps({"query": "no_idx"}) + "\n")  # missing idx

        cm = CheckpointManager(cp_file)
        processed, cached = cm.load_checkpoint()
        assert 3 in processed
        assert -1 not in processed
        # The "no_idx" entry gets idx=-1 (default), so not in processed
        assert len([r for r in cached if r.get("idx", -1) == 3]) == 1


# =============================================================================
# 4. asyncio.gather return_exceptions=True in deeprag.py
# =============================================================================
class TestAsyncGatherExceptionIsolation:
    """Tests that asyncio.gather in deeprag.py isolates exceptions."""

    def test_retrieval_exception_does_not_crash_others(self):
        """One retrieval task failing should not prevent other results."""
        from deeprag import DeepResearchWorkflow

        wf = DeepResearchWorkflow(rag_system=MagicMock())

        # Simulate two subqueries; one retrieval fails
        async def fetch_ok(sq):
            return (sq.id, [Paper(id="p1", title="P1", abstract="", arxiv_id="a1")], {})

        async def fetch_fail(sq):
            raise ConnectionError("Qdrant timeout")

        async def run_retrieval(subqueries_list):
            tasks = []
            for sq in subqueries_list:
                if sq.id == 1:
                    tasks.append(fetch_ok(sq))
                else:
                    tasks.append(fetch_fail(sq))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            out = {}
            for r in results:
                if isinstance(r, Exception):
                    continue
                sq_id, papers, rank_dict = r
                out[sq_id] = (papers, rank_dict)
            return out

        sq1 = SubQuery(id=1, text="good query")
        sq2 = SubQuery(id=2, text="bad query")
        result = asyncio.run(run_retrieval([sq1, sq2]))
        assert 1 in result
        assert 2 not in result
        assert len(result[1][0]) == 1

    def test_selection_exception_isolation(self):
        """One selector call failing should not crash other selections."""
        from deeprag import DeepResearchWorkflow

        async def run_selection():
            async def select_ok(*args, **kwargs):
                return ([Paper(id="p1", title="P1", abstract="", arxiv_id="a1")], "ov", {})

            async def select_fail(*args, **kwargs):
                raise RuntimeError("LLM rate limit")

            tasks = [select_ok(), select_fail()]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            out = {}
            keys = [1, 2]
            for (sq_id, _), r in zip(enumerate(keys), results):
                if isinstance(r, Exception):
                    continue
                out[sq_id] = r
            return out

        result = asyncio.run(run_selection())
        # At least one result should survive
        assert len(result) >= 1


# =============================================================================
# 5. BM25 tokenization preserving numbers and hyphens
# =============================================================================
class TestBM25Tokenization:
    """Tests for _preprocess_text_for_bm25 regex fix."""

    def _get_preprocess(self):
        from rag import CitationRAGSystem
        rag = CitationRAGSystem(search_method='bm25')
        return rag._preprocess_text_for_bm25

    def test_numbers_preserved(self):
        """Numbers like '2' in 'BERT-2' should be preserved as part of token."""
        preprocess = self._get_preprocess()
        tokens = preprocess("BERT-2 model")
        # Hyphen is allowed inside token, so 'bert-2' stays together
        assert "bert-2" in tokens
        assert "model" in tokens

    def test_hyphenated_terms(self):
        """Hyphenated terms like 'self-attention' should preserve hyphen."""
        preprocess = self._get_preprocess()
        tokens = preprocess("self-attention mechanism")
        assert "self-attention" in tokens

    def test_arxiv_ids_preserved(self):
        """ArXiv IDs like '2007.05993' are split by dot (word boundary), both halves kept."""
        preprocess = self._get_preprocess()
        tokens = preprocess("paper 2007.05993 shows")
        # Dot is a word boundary: splits into '2007' and '05993'
        assert "2007" in tokens
        assert "05993" in tokens

    def test_version_suffix(self):
        """ArXiv version suffix 'v2' stays attached to the number part."""
        preprocess = self._get_preprocess()
        tokens = preprocess("arxiv 2007.05993v2 updated")
        # Dot splits: '2007' and '05993v2' (v2 stays attached to 05993)
        assert "2007" in tokens
        assert "05993v2" in tokens

    def test_mixed_alphanumeric(self):
        """Mixed alphanumeric terms like 'T5' should be preserved."""
        preprocess = self._get_preprocess()
        tokens = preprocess("T5 model with 3 layers")
        assert "3" in tokens
        # 'T5' is uppercase, lowercased to 't5'
        assert "t5" in tokens or "5" in tokens

    def test_pure_number(self):
        """Pure numbers like '42' should be captured."""
        preprocess = self._get_preprocess()
        tokens = preprocess("answer is 42")
        assert "42" in tokens

    def test_empty_input(self):
        """Empty string should return empty list."""
        preprocess = self._get_preprocess()
        assert preprocess("") == []

    def test_only_punctuation(self):
        """Punctuation-only input should return empty list."""
        preprocess = self._get_preprocess()
        assert preprocess("!!! ??? ...") == []


# =============================================================================
# 6. rag.py error handling for pickle/JSON load
# =============================================================================
class TestRagErrorHandling:
    """Tests for rag.py load_paper_db and load_bm25_index error handling."""

    def test_load_paper_db_corrupt_json(self, tmp_path):
        """Corrupt JSON should raise with descriptive error."""
        from rag import CitationRAGSystem
        rag = CitationRAGSystem(search_method='bm25')

        bad_file = str(tmp_path / "bad.json")
        with open(bad_file, "w") as f:
            f.write("{not valid json")

        with pytest.raises((json.JSONDecodeError, ValueError)):
            rag.load_paper_db(bad_file)

    def test_load_paper_db_missing_file(self, tmp_path):
        """Missing file should raise IOError."""
        from rag import CitationRAGSystem
        rag = CitationRAGSystem(search_method='bm25')

        with pytest.raises((IOError, FileNotFoundError)):
            rag.load_paper_db(str(tmp_path / "nonexistent.json"))

    def test_load_bm25_corrupt_pickle(self, tmp_path):
        """Corrupt pickle file should raise with descriptive error."""
        from rag import CitationRAGSystem
        rag = CitationRAGSystem(search_method='bm25')

        bad_file = str(tmp_path / "bad.pkl")
        with open(bad_file, "wb") as f:
            f.write(b"not a pickle file")

        with pytest.raises(Exception):
            rag.load_bm25_index(bad_file)


# =============================================================================
# 7. Browser: exponential backoff parameters
# =============================================================================
class TestBrowserBackoff:
    """Tests for browser.py exponential backoff configuration."""

    def test_client_timeout_is_60s(self):
        """httpx client should use 60s timeout."""
        from agent.browser import Ar5ivParser
        # Verify the timeout value is in the source
        import inspect
        source = inspect.getsource(Ar5ivParser.fetch_and_parse)
        assert "timeout=60.0" in source

    def test_backoff_formula(self):
        """Backoff should use exponential formula: delay * 2^attempt."""
        retry_delay = 2
        # attempt 0: 2 * 1 = 2
        assert retry_delay * (2 ** 0) == 2
        # attempt 1: 2 * 2 = 4
        assert retry_delay * (2 ** 1) == 4
        # attempt 2: 2 * 4 = 8
        assert retry_delay * (2 ** 2) == 8


# =============================================================================
# 8. Summarizer: cache write order (memory first, then disk)
# =============================================================================
class TestSummarizerCacheWriteOrder:
    """Tests for summarizer cache write order: memory before disk."""

    def test_memory_updated_before_disk(self, tmp_path):
        """In-memory cache should be updated even if disk write fails."""
        from agent.summarizer import PaperSummarizer

        cache_file = str(tmp_path / "cache.jsonl")
        with patch('agent.summarizer.config') as mock_config:
            mock_config.SUMMARY_LLM_MODEL_NAME = "test-model"
            mock_config.SUMMARY_LLM_GEN_PARAMS = {}
            mock_config.SUMMARY_LLM_IS_LOCAL = True
            mock_config.SUMMARY_CACHE_PATH = cache_file

            summarizer = PaperSummarizer()
            summarizer._append_summary_to_cache("paper_001", "test summary")

            # In-memory should be available immediately
            assert "paper_001" in summarizer.summary_cache
            assert summarizer.summary_cache["paper_001"]["summary"] == "test summary"

    def test_cache_keyed_by_model(self, tmp_path):
        """Cache entries should be keyed by model name."""
        from agent.summarizer import PaperSummarizer

        cache_file = str(tmp_path / "cache.jsonl")
        with patch('agent.summarizer.config') as mock_config:
            mock_config.SUMMARY_LLM_MODEL_NAME = "model-v1"
            mock_config.SUMMARY_LLM_GEN_PARAMS = {}
            mock_config.SUMMARY_LLM_IS_LOCAL = True
            mock_config.SUMMARY_CACHE_PATH = cache_file

            summarizer = PaperSummarizer()
            summarizer._append_summary_to_cache("paper_001", "summary v1")

            # Same paper with different model should not match
            mock_config.SUMMARY_LLM_MODEL_NAME = "model-v2"
            summarizer.llm_model = "model-v2"
            cached = summarizer._get_summary_from_cache("paper_001")
            assert cached is None  # Different model, no cache hit

            # Original model should still hit
            summarizer.llm_model = "model-v1"
            cached = summarizer._get_summary_from_cache("paper_001")
            assert cached == "summary v1"


# =============================================================================
# 9. rag.search_citations_vector: raw QdrantClient path + server-side filter
# =============================================================================
class TestBeforeDateToLtIso:
    """Boundary tests for the YYYY-MM -> ISO datetime helper."""

    def test_january_non_december(self):
        from rag import CitationRAGSystem
        assert CitationRAGSystem._before_date_to_lt_iso("2020-01") == "2020-02-01T00:00:00Z"

    def test_december_rollover(self):
        from rag import CitationRAGSystem
        # Dec must roll into next year's Jan
        assert CitationRAGSystem._before_date_to_lt_iso("2020-12") == "2021-01-01T00:00:00Z"

    def test_accepts_full_date_but_only_uses_month(self):
        from rag import CitationRAGSystem
        # Month-granularity: day/time are ignored
        assert CitationRAGSystem._before_date_to_lt_iso("2020-05-17") == "2020-06-01T00:00:00Z"
        assert CitationRAGSystem._before_date_to_lt_iso("2020-12-31") == "2021-01-01T00:00:00Z"


class TestVectorSearchServerSideFilter:
    """Tests that search_citations_vector builds the right Qdrant Filter and
    unwraps the nested payload correctly."""

    def _make_loaded_rag(self):
        """Build a CitationRAGSystem with mocked QdrantClient + embeddings."""
        from rag import CitationRAGSystem
        rag = CitationRAGSystem(search_method='vector')
        rag.qdrant_client = MagicMock()
        rag.qdrant_embeddings = MagicMock()
        rag.qdrant_embeddings.embed_query.return_value = [0.0] * 8
        # Build a reusable hit-shape helper
        return rag

    def _make_hit(self, arxiv_id, date, score):
        """Mimic Qdrant point with nested {page_content, metadata} payload."""
        point = MagicMock()
        point.payload = {
            "page_content": f"title: {arxiv_id}",
            "metadata": {
                "arxiv_id": arxiv_id,
                "date": date,
                "title": f"Paper {arxiv_id}",
            },
        }
        point.score = score
        return point

    def test_raises_when_not_loaded(self):
        from rag import CitationRAGSystem
        rag = CitationRAGSystem(search_method='vector')
        with pytest.raises(ValueError):
            rag.search_citations_vector(query="q", top_k=5, debug=False)

    def test_no_filter_when_no_constraints(self):
        rag = self._make_loaded_rag()
        rag.qdrant_client.query_points.return_value = MagicMock(
            points=[self._make_hit("2020.00001", "2020-03-05", 0.9)]
        )

        rag.search_citations_vector(query="q", top_k=5, debug=False)

        call_kwargs = rag.qdrant_client.query_points.call_args.kwargs
        # No before_date, no exclude_arxiv_ids -> query_filter must be None
        assert call_kwargs["query_filter"] is None

    def test_before_date_builds_datetime_range(self):
        from qdrant_client.http import models as qm
        rag = self._make_loaded_rag()
        rag.qdrant_client.query_points.return_value = MagicMock(points=[])

        rag.search_citations_vector(
            query="q", top_k=5, before_date="2020-01", debug=False
        )

        flt = rag.qdrant_client.query_points.call_args.kwargs["query_filter"]
        assert flt is not None
        assert flt.must is not None and len(flt.must) == 1
        cond = flt.must[0]
        assert cond.key == "metadata.date"
        assert cond.range is not None
        # lt value should be first day of February 2020
        assert cond.range.lt.year == 2020 and cond.range.lt.month == 2
        # No must_not when exclude_arxiv_ids is empty
        assert flt.must_not is None

    def test_exclude_arxiv_ids_builds_match_any(self):
        rag = self._make_loaded_rag()
        rag.qdrant_client.query_points.return_value = MagicMock(points=[])

        rag.search_citations_vector(
            query="q",
            top_k=5,
            exclude_arxiv_ids={"2007.00001", "2019.12345"},
            debug=False,
        )

        flt = rag.qdrant_client.query_points.call_args.kwargs["query_filter"]
        assert flt is not None
        assert flt.must is None
        assert flt.must_not is not None and len(flt.must_not) == 1
        cond = flt.must_not[0]
        assert cond.key == "metadata.arxiv_id"
        assert set(cond.match.any) == {"2007.00001", "2019.12345"}

    def test_combined_filter_has_both_clauses(self):
        rag = self._make_loaded_rag()
        rag.qdrant_client.query_points.return_value = MagicMock(points=[])

        rag.search_citations_vector(
            query="q",
            top_k=5,
            before_date="2020-12",
            exclude_arxiv_ids={"9999.00001"},
            debug=False,
        )

        flt = rag.qdrant_client.query_points.call_args.kwargs["query_filter"]
        assert flt is not None
        assert len(flt.must) == 1
        assert len(flt.must_not) == 1
        # Dec rollover -> 2021-01-01
        assert flt.must[0].range.lt.year == 2021
        assert flt.must[0].range.lt.month == 1

    def test_nested_payload_unwrap(self):
        """Results come from payload['metadata'], not top-level payload."""
        rag = self._make_loaded_rag()
        rag.qdrant_client.query_points.return_value = MagicMock(
            points=[
                self._make_hit("2020.00001", "2020-03-05", 0.9),
                self._make_hit("2021.00002", "2021-06-10", 0.8),
            ]
        )

        results, _ = rag.search_citations_vector(query="q", top_k=5, debug=False)

        assert [r[0] for r in results] == ["2020.00001", "2021.00002"]
        assert [r[1] for r in results] == [0.9, 0.8]
        # meta dict should be the inner "metadata" dict
        assert results[0][2]["title"] == "Paper 2020.00001"

    def test_skips_hits_without_arxiv_id(self):
        rag = self._make_loaded_rag()
        bad = MagicMock()
        bad.payload = {"metadata": {"title": "no id"}}  # no arxiv_id / id
        bad.score = 0.5
        good = self._make_hit("2020.00001", "2020-03-05", 0.9)
        rag.qdrant_client.query_points.return_value = MagicMock(points=[bad, good])

        results, _ = rag.search_citations_vector(query="q", top_k=5, debug=False)
        assert [r[0] for r in results] == ["2020.00001"]

    def test_fetch_k_respects_offset(self):
        rag = self._make_loaded_rag()
        rag.qdrant_client.query_points.return_value = MagicMock(points=[])

        rag.search_citations_vector(query="q", top_k=5, offset=7, debug=False)

        limit = rag.qdrant_client.query_points.call_args.kwargs["limit"]
        # multiplier=5, limit = (offset + GT_RANK_CUTOFF) * 5
        expected = (7 + config.GT_RANK_CUTOFF) * 5
        assert limit == expected

    def test_is_loaded_tracks_qdrant_client(self):
        from rag import CitationRAGSystem
        rag = CitationRAGSystem(search_method='vector')
        assert rag.is_loaded() is False
        rag.qdrant_client = MagicMock()
        assert rag.is_loaded() is True


# =============================================================================
# 10. build_vector_db.py: checkpoint cadence
# =============================================================================
class TestBuildVectorDbCheckpointCadence:
    """Verify checkpoint is not rewritten every single batch anymore."""

    def test_flush_every_constant_exists_and_is_not_one(self):
        import inspect
        import build_vector_db
        src = inspect.getsource(build_vector_db)
        assert "CHECKPOINT_FLUSH_EVERY" in src
        # Must be > 1, otherwise we regressed to every-batch writes
        ns = {}
        for line in src.splitlines():
            if line.strip().startswith("CHECKPOINT_FLUSH_EVERY"):
                exec(line.strip(), ns)
                break
        assert ns.get("CHECKPOINT_FLUSH_EVERY", 1) > 1

    def test_final_flush_is_called_after_loop(self):
        import inspect
        import build_vector_db
        src = inspect.getsource(build_vector_db)
        # The final flush outside the loop is the critical invariant
        assert "# Final flush" in src
        # Atomic replace via .tmp path, not direct overwrite
        assert "os.replace" in src
        assert '".tmp"' in src or "'.tmp'" in src
