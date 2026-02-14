# =============================================================================
# ScholarGym Configuration
# =============================================================================

# --- Service Endpoints ---
QDRANT_URL = "http://localhost:6433"
OLLAMA_URL = "http://localhost:11434"

# --- Data Paths ---
PAPER_DB_PATH = 'data/scholargym_paper_db.json'
BENCHMARK_PATH = 'data/scholargym_bench.jsonl'
EVAL_BASE_DIR = 'eval_results'

# --- Index Paths ---
FAISS_PATH_PREFIX = 'data/faiss_index'
BM25_PATH = 'data/bm25_index.pkl'
DEFAULT_SEARCH_METHOD = 'bm25'

# --- Model Configurations ---
EMBEDDING_MODEL_PATH = 'Qwen/Qwen3-Embedding-0.6B'
DEVICE = 'cuda:0'

LLM_MODEL_NAME = 'qwen3:8b'
IS_LOCAL_LLM = True

LLM_GEN_PARAMS = {
    "max_tokens": 8192,
    "temperature": 0,
    "top_p": 1,
    "stream": False
}

BROWSER_MAX_TOKENS = 8192

# --- LLM Advanced Features ---
ENABLE_REASONING = True
ENABLE_STRUCTURED_OUTPUT = False
SAVE_AGENT_TRACES = False
PLANNER_ABLATION = False

# --- RAG System Configurations ---
EMBEDDING_BATCH_SIZE = 256
VECTOR_SEARCH_TOP_K = 10
BM25_SEARCH_TOP_K = 10
HYBRID_SEARCH_TOP_K = 10
HYBRID_VECTOR_WEIGHT = 0.5
HYBRID_BM25_WEIGHT = 0.5

# --- Evaluation Configurations ---
EVAL_TOP_K_VALUES = [5, 10, 20]
EVAL_PROMPT_TYPE = 'complex'
EVAL_SEARCH_METHOD = 'bm25'
EVAL_WORKFLOW = 'deep_research'
EVAL_MAX_ITERATIONS = 5

# --- Deep Research Configurations ---
CONTEXT_MAX_LENGTH_CHARS = 32768
ENABLE_LLM_FILTERING = True
LLM_FILTERING_BATCH_SIZE = 5
MAX_RESULTS_PER_QUERY = 10
MAX_PAGES_PER_QUERY = 3
ENABLE_SUMMARIZATION = False
BROWSER_MODE = 'NONE'  # Options: 'PRE_ENRICH', 'REFRESH', 'INCREMENTAL', 'NONE'
SUMMARY_CACHE_PATH = 'summary_cache.jsonl'
SUMMARY_ABSTRACT_CHAR_THRESHOLD = 20000

# --- Evaluation Metrics ---
GT_RANK_CUTOFF = 100

# --- Summarization Model ---
SUMMARY_LLM_MODEL_NAME = 'qwen3:8b'
SUMMARY_LLM_IS_LOCAL = True
SUMMARY_LLM_GEN_PARAMS = {
    "max_tokens": 4096,
    "temperature": 0.2,
    "top_p": 0.95,
    "stream": False
}

DEBUG = False
CASE_STUDY_OUTPUT_DIR = './case_study'

# --- Paper Corpus ---
TOTAL_PAPER_NUM = 570205
