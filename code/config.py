# =============================================================================
# ScholarGym Configuration
#
# Edit the fields in "Required / commonly changed" first.
# Most other fields can be left at their defaults.
#
# This file is the canonical base configuration for ScholarGym:
#   - manual single runs can edit this file directly
#   - eval.py loads this first, then applies `--config` / CLI overrides
#   - scripts/exp/launcher.py uses it as the template for generated
#     `runs/<name>/config.py` snapshots
#
# Flat module-level constants. Most entry points do `import config` and read
# attributes directly. CLI args in eval.py override a subset of these.
# =============================================================================


# =============================================================================
# Required / commonly changed
# =============================================================================

# Change this to the model you want to evaluate.
LLM_MODEL_NAME = 'qwen3-8b'

# Set True for local serving backends like Ollama or sglang.
IS_LOCAL_LLM = True

# Benchmark file to evaluate.
BENCHMARK_PATH = 'data/scholargym_bench.jsonl'

# Retrieval method: 'bm25' or 'vector'.
EVAL_SEARCH_METHOD = 'bm25'

# Workflow: 'simple' or 'deep_research'.
EVAL_WORKFLOW = 'deep_research'


# =============================================================================
# Services
# =============================================================================

# Qdrant service endpoint.
QDRANT_URL = "http://localhost:6433"

# Local Ollama endpoint.
OLLAMA_URL = "http://localhost:11434"


# =============================================================================
# Data paths
# =============================================================================

# Paper database JSON path.
PAPER_DB_PATH = 'data/scholargym_paper_db.json'

# Base output directory for legacy eval.py runs.
EVAL_BASE_DIR = 'eval_results'


# =============================================================================
# Retrieval / indexes
# =============================================================================

# BM25 index path.
BM25_PATH = 'data/bm25_index.pkl'

# Default retrieval backend outside explicit eval overrides.
DEFAULT_SEARCH_METHOD = 'bm25'

# Qdrant collection name.
QDRANT_COLLECTION_NAME = "paper_knowledge_base"

# Embedding model used for Qdrant indexing/querying.
QDRANT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# Vector retrieval fanout before later filtering.
VECTOR_SEARCH_TOP_K = 10

# BM25 retrieval fanout before later filtering.
BM25_SEARCH_TOP_K = 10


# =============================================================================
# LLM — model + generation params
# =============================================================================

# Compute device for local components.
DEVICE = 'cuda:0'

# Generation parameters for the main evaluator model.
LLM_GEN_PARAMS = {
    "max_tokens": 8192,
    "temperature": 0,
    "top_p": 1,
    "stream": False,
}

# Max tokens when browser-related LLM calls are made.
BROWSER_MAX_TOKENS = 8192

# Usually keep default unless comparing reasoning ablations.
ENABLE_REASONING = True

# Use structured outputs when the provider supports them.
ENABLE_STRUCTURED_OUTPUT = False

# Persist detailed agent traces for debugging.
SAVE_AGENT_TRACES = False

# Enable planner ablation mode for experiments.
PLANNER_ABLATION = False


# =============================================================================
# Evaluation knobs
# =============================================================================

# Top-k list used by the simple workflow.
EVAL_TOP_K_VALUES = [5, 10, 20]

# Prompt template family.
EVAL_PROMPT_TYPE = 'complex'

# Iteration count for deep_research.
EVAL_MAX_ITERATIONS = 5

# Ground-truth rank cutoff for metrics.
GT_RANK_CUTOFF = 100


# =============================================================================
# Deep Research workflow
# =============================================================================

# Context budget for long prompts.
CONTEXT_MAX_LENGTH_CHARS = 32768

# Whether to run selector-side LLM filtering.
ENABLE_LLM_FILTERING = True

# Batch size for selector filtering.
LLM_FILTERING_BATCH_SIZE = 5

# Results retrieved per subquery in deep_research.
MAX_RESULTS_PER_QUERY = 10

# Browser page cap per subquery.
MAX_PAGES_PER_QUERY = 3

# Enable abstract/content summarization.
ENABLE_SUMMARIZATION = False

# Browser mode: 'NONE', 'PRE_ENRICH', 'REFRESH', or 'INCREMENTAL'.
BROWSER_MODE = 'NONE'

# JSONL cache for summarizer outputs.
SUMMARY_CACHE_PATH = 'summary_cache.jsonl'

# Summarize abstracts longer than this threshold.
SUMMARY_ABSTRACT_CHAR_THRESHOLD = 20000


# =============================================================================
# Summarization model
# =============================================================================

# Model used by the summarizer.
SUMMARY_LLM_MODEL_NAME = 'qwen3-8b'

# Whether summarizer model is served locally.
SUMMARY_LLM_IS_LOCAL = True

# Generation parameters for the summarizer model.
SUMMARY_LLM_GEN_PARAMS = {
    "max_tokens": 4096,
    "temperature": 0.2,
    "top_p": 0.95,
    "stream": False,
}


# =============================================================================
# Debug / tracing
# =============================================================================

# Enable extra debug behaviors.
DEBUG = False

# Print more intermediate workflow details.
VERBOSE = False

# Directory for case-study artifacts.
CASE_STUDY_OUTPUT_DIR = './case_study'


# =============================================================================
# Paper corpus
# =============================================================================

# Total number of papers in the corpus.
TOTAL_PAPER_NUM = 570205
