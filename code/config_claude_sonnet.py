# Config for Claude Sonnet 4.6 evaluation
from config import *

LLM_MODEL_NAME = 'claude-sonnet-4-6'
IS_LOCAL_LLM = False
BENCHMARK_PATH = 'data/scholargym_bench.jsonl'
EVAL_MAX_ITERATIONS = 5
MAX_RESULTS_PER_QUERY = 10
BROWSER_MODE = 'NONE'
ENABLE_REASONING = False
ENABLE_STRUCTURED_OUTPUT = False
