"""Shared fixtures and sys.path setup for ScholarGym tests."""
import sys
import os
import json
import types
import pytest

# Add code/ and scripts/ to sys.path so we can import project modules
CODE_DIR = os.path.join(os.path.dirname(__file__), '..')
if CODE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(CODE_DIR))

# scripts/ is at repo root (one level above code/)
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))

# Provide a minimal config module before any project import touches it
_config = types.ModuleType("config")
_config.QDRANT_URL = "http://localhost:6433"
_config.OLLAMA_URL = "http://127.0.0.1:8001"
_config.PAPER_DB_PATH = "data/scholargym_paper_db.json"
_config.BENCHMARK_PATH = "data/scholargym_bench_short.jsonl"
_config.EVAL_BASE_DIR = "eval_results"
_config.BM25_PATH = "data/bm25_index.pkl"
_config.DEFAULT_SEARCH_METHOD = "bm25"
_config.QDRANT_COLLECTION_NAME = "paper_knowledge_base"
_config.QDRANT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
_config.DEVICE = "cpu"
_config.LLM_MODEL_NAME = "test-model"
_config.IS_LOCAL_LLM = True
_config.LLM_GEN_PARAMS = {"max_tokens": 128, "temperature": 0, "top_p": 1, "stream": False}
_config.BROWSER_MAX_TOKENS = 8192
_config.ENABLE_REASONING = False
_config.ENABLE_STRUCTURED_OUTPUT = False
_config.SAVE_AGENT_TRACES = False
_config.PLANNER_ABLATION = False
_config.VECTOR_SEARCH_TOP_K = 10
_config.BM25_SEARCH_TOP_K = 10
_config.EVAL_TOP_K_VALUES = [5, 10, 20]
_config.EVAL_PROMPT_TYPE = "complex"
_config.EVAL_SEARCH_METHOD = "bm25"
_config.EVAL_WORKFLOW = "deep_research"
_config.EVAL_MAX_ITERATIONS = 5
_config.CONTEXT_MAX_LENGTH_CHARS = 32768
_config.ENABLE_LLM_FILTERING = True
_config.LLM_FILTERING_BATCH_SIZE = 5
_config.MAX_RESULTS_PER_QUERY = 10
_config.MAX_PAGES_PER_QUERY = 3
_config.ENABLE_SUMMARIZATION = False
_config.BROWSER_MODE = "NONE"
_config.SUMMARY_CACHE_PATH = "summary_cache.jsonl"
_config.SUMMARY_ABSTRACT_CHAR_THRESHOLD = 20000
_config.GT_RANK_CUTOFF = 100
_config.SUMMARY_LLM_MODEL_NAME = "test-model"
_config.SUMMARY_LLM_IS_LOCAL = True
_config.SUMMARY_LLM_GEN_PARAMS = {"max_tokens": 128, "temperature": 0, "top_p": 1, "stream": False}
_config.DEBUG = False
_config.VERBOSE = False
_config.CASE_STUDY_OUTPUT_DIR = "./case_study"
_config.TOTAL_PAPER_NUM = 570205
sys.modules["config"] = _config


# ---------------------------------------------------------------------------
# Fixture file loading helpers
# ---------------------------------------------------------------------------
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def load_fixture(filename):
    """Load a JSON fixture file from the fixtures directory."""
    path = os.path.join(FIXTURES_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Benchmark data fixtures (loaded from benchmark_samples.json)
# ---------------------------------------------------------------------------
@pytest.fixture
def benchmark_fixtures():
    """Raw benchmark fixture data."""
    return load_fixture("benchmark_samples.json")


@pytest.fixture
def sample_benchmark_data(benchmark_fixtures):
    """Synthetic benchmark data for tests (first 2 entries for backward compat)."""
    entries = benchmark_fixtures["benchmark_entries"]
    # Return entries without the 'id' key (test code expects standard benchmark format)
    result = []
    for entry in entries:
        e = {k: v for k, v in entry.items() if k != "id"}
        result.append(e)
    return result


@pytest.fixture
def gt_ids_a(benchmark_fixtures):
    """Ground truth IDs set A — 3 papers."""
    return set(benchmark_fixtures["gt_id_sets"]["set_a"])


@pytest.fixture
def gt_ids_b(benchmark_fixtures):
    """Ground truth IDs set B — 5 papers."""
    return set(benchmark_fixtures["gt_id_sets"]["set_b"])


@pytest.fixture
def gt_ids_single(benchmark_fixtures):
    """Ground truth IDs — single paper."""
    return set(benchmark_fixtures["gt_id_sets"]["single"])


@pytest.fixture
def gt_ids_large(benchmark_fixtures):
    """Ground truth IDs — 10 papers."""
    return set(benchmark_fixtures["gt_id_sets"]["large"])


# ---------------------------------------------------------------------------
# Metrics test case fixtures (loaded from metrics_cases.json)
# ---------------------------------------------------------------------------
@pytest.fixture
def metrics_fixtures():
    """All metrics test case data."""
    return load_fixture("metrics_cases.json")


# ---------------------------------------------------------------------------
# Utils test case fixtures (loaded from utils_cases.json)
# ---------------------------------------------------------------------------
@pytest.fixture
def utils_fixtures():
    """All utility function test case data."""
    return load_fixture("utils_cases.json")


# ---------------------------------------------------------------------------
# DeepRAG test case fixtures (loaded from deeprag_cases.json)
# ---------------------------------------------------------------------------
@pytest.fixture
def deeprag_fixtures():
    """All DeepRAG workflow test case data."""
    return load_fixture("deeprag_cases.json")


# ---------------------------------------------------------------------------
# Factory fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def make_benchmark_entry():
    """Factory fixture: generate benchmark data entries on demand."""
    def _make(query="test query", n_gt=2, n_noise=1, qid="test_0"):
        gt_papers = [
            {"arxiv_id": f"gt_{i:05d}", "title": f"GT Paper {i}", "year": 2020}
            for i in range(n_gt)
        ]
        noise_papers = [
            {"arxiv_id": f"noise_{i:05d}", "title": f"Noise {i}", "year": 2020}
            for i in range(n_noise)
        ]
        all_papers = gt_papers + noise_papers
        labels = [1] * n_gt + [0] * n_noise
        return {
            "query": query,
            "cited_paper": all_papers,
            "gt_label": labels,
            "date": "2020-09",
            "source": "test",
            "qid": qid,
            "valid": True,
        }
    return _make


@pytest.fixture
def make_paper():
    """Factory fixture: generate Paper objects on demand."""
    from structures import Paper
    _counter = [0]

    def _make(arxiv_id=None, **kwargs):
        _counter[0] += 1
        aid = arxiv_id or f"auto_{_counter[0]:05d}"
        defaults = {
            "id": f"p{_counter[0]}",
            "title": f"Paper {_counter[0]}",
            "abstract": "abs",
            "arxiv_id": aid,
        }
        defaults.update(kwargs)
        return Paper(**defaults)
    return _make
