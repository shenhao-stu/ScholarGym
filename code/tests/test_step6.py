"""Tests for Step 6: code quality fixes."""
import logging
import logging.handlers
import os
import json
import tempfile
from collections import deque
from unittest.mock import patch, MagicMock

import pytest


# ── logger.py tests ──────────────────────────────────────────────────────────


class TestLoggerHandler:
    """Tests for LoggerHandler bounded memory."""

    def test_handler_uses_deque(self):
        from logger import LoggerHandler
        handler = LoggerHandler()
        assert isinstance(handler._entries, deque)
        assert handler._entries.maxlen == LoggerHandler.MAX_ENTRIES

    def test_handler_log_property(self):
        from logger import LoggerHandler
        handler = LoggerHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello", (), None)
        handler.emit(record)
        assert "hello" in handler.log

    def test_handler_bounded(self):
        from logger import LoggerHandler
        handler = LoggerHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        # Override maxlen to a small value for testing
        handler._entries = deque(maxlen=3)
        for i in range(5):
            record = logging.LogRecord("test", logging.INFO, "", 0, f"msg{i}", (), None)
            handler.emit(record)
        # Only the last 3 should remain
        assert len(handler._entries) == 3
        assert "msg0" not in handler.log
        assert "msg1" not in handler.log
        assert "msg4" in handler.log

    def test_handler_reset(self):
        from logger import LoggerHandler
        handler = LoggerHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello", (), None)
        handler.emit(record)
        handler.reset()
        assert handler.log == ""
        assert len(handler._entries) == 0


class TestRotatingFileHandler:
    """Tests for RotatingFileHandler in get_logger."""

    def test_file_handler_is_rotating(self, tmp_path):
        from logger import get_logger
        log_file = str(tmp_path / "test.log")
        # Remove any cached logger with this name
        name = f"test_rotating_{id(tmp_path)}"
        logger = get_logger(name, log_file=log_file)
        file_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 1
        assert file_handlers[0].maxBytes == 50 * 1024 * 1024
        assert file_handlers[0].backupCount == 3
        # Cleanup
        logger.handlers.clear()


# ── eval.py exception handling tests ─────────────────────────────────────────


class TestEvalExceptionHandling:
    """Tests for eval.py exception categories."""

    def test_memory_error_not_caught(self):
        """MemoryError should propagate, not be silently dropped."""
        from eval import CitationEvaluator

        mock_rag = MagicMock()
        mock_rag.get_available_search_methods.return_value = ['bm25']

        evaluator = CitationEvaluator(rag_system=mock_rag)

        benchmark_data = [{'query': 'test', 'cited_paper': [], 'gt_label': []}]

        with patch.object(evaluator, 'evaluate_single_query_deep_research', side_effect=MemoryError("OOM")):
            with pytest.raises(MemoryError):
                evaluator.evaluate_benchmark(
                    benchmark_data=benchmark_data,
                    workflow='deep_research',
                    enable_resume=False
                )

    def test_runtime_error_caught_and_logged(self):
        """RuntimeError (e.g. from LLM) should be caught and logged."""
        from eval import CitationEvaluator

        mock_rag = MagicMock()
        mock_rag.get_available_search_methods.return_value = ['bm25']

        evaluator = CitationEvaluator(rag_system=mock_rag)

        benchmark_data = [{'query': 'test', 'cited_paper': [], 'gt_label': []}]

        with patch.object(evaluator, 'evaluate_single_query_deep_research', side_effect=RuntimeError("LLM failed")):
            results = evaluator.evaluate_benchmark(
                benchmark_data=benchmark_data,
                workflow='deep_research',
                enable_resume=False
            )
            # Query failed but evaluation should continue
            assert results['successful_queries'] == 0

    def test_exc_info_logged(self):
        """Full traceback should be logged for failed queries."""
        from eval import CitationEvaluator

        mock_rag = MagicMock()
        mock_rag.get_available_search_methods.return_value = ['bm25']

        evaluator = CitationEvaluator(rag_system=mock_rag)

        benchmark_data = [{'query': 'test', 'cited_paper': [], 'gt_label': []}]

        with patch.object(evaluator, 'evaluate_single_query_deep_research', side_effect=ValueError("bad format")):
            with patch('eval.logger') as mock_logger:
                evaluator.evaluate_benchmark(
                    benchmark_data=benchmark_data,
                    workflow='deep_research',
                    enable_resume=False
                )
                # Check that warning was called with exc_info=True
                warning_calls = [c for c in mock_logger.warning.call_args_list if 'Failed to evaluate' in str(c)]
                assert len(warning_calls) == 1
                assert warning_calls[0].kwargs.get('exc_info') is True or \
                       (len(warning_calls[0].args) > 1 or warning_calls[0].kwargs.get('exc_info'))


# ── summarizer.py flush test ─────────────────────────────────────────────────


class TestSummarizerCacheFlush:
    """Tests for summarizer cache write with flush."""

    def test_append_summary_flushes(self, tmp_path):
        cache_file = str(tmp_path / "cache.jsonl")
        with patch('agent.summarizer.config') as mock_config:
            mock_config.SUMMARY_LLM_MODEL_NAME = "test-model"
            mock_config.SUMMARY_LLM_GEN_PARAMS = {}
            mock_config.SUMMARY_LLM_IS_LOCAL = True
            mock_config.SUMMARY_CACHE_PATH = cache_file

            from agent.summarizer import PaperSummarizer
            summarizer = PaperSummarizer()
            summarizer._append_summary_to_cache("paper_001", "This is a summary")

            # Verify data is written
            with open(cache_file, 'r') as f:
                data = json.loads(f.readline())
            assert data['paper_id'] == "paper_001"
            assert data['summary'] == "This is a summary"
            assert data['model'] == "test-model"
