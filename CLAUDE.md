# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ScholarGym is a benchmark framework for evaluating LLMs' capabilities in the information-gathering stage of deep research on academic literature. It evaluates 2,536 expert-annotated queries against a static corpus of 570K arXiv papers using deterministic retrieval (BM25/vector/hybrid).

## Running Evaluation

```bash
# Install
conda create -n scholargym python=3.10
conda activate scholargym
pip install -r requirements.txt

# Run evaluation (all source files are in code/, run from project root)
python code/eval.py \
    --paper_db data/scholargym_paper_db.json \
    --benchmark_jsonl data/scholargym_bench.jsonl \
    --bm25_path data/bm25_index.pkl \
    --workflow deep_research \
    --search_method bm25 \
    --max_iterations 5 \
    --results_per_query 10 \
    --llm_model qwen3:8b

# Key CLI flags
#   --workflow: "simple" (single-pass query expansion) or "deep_research" (multi-iteration agents)
#   --search_method: "bm25", "vector", or "hybrid"
#   --browser_mode: "NONE", "PRE_ENRICH", "REFRESH", "INCREMENTAL"
#   --is_local: use local Ollama endpoint
#   --config: path to custom config.py (overrides default)
```

## Architecture

### Entry Point & Configuration
- `code/eval.py` — Main entry point. `CitationEvaluator` orchestrates workflows, collects metrics, saves results.
- `code/config.py` — All configuration as module-level variables. CLI args override these. `eval.py` supports dynamic config loading via `--config`.

### Two Workflows
- **SimpleWorkflow** (`code/simplerag.py`) — Single-pass: LLM generates diverse query keys, retrieves papers, deduplicates.
- **DeepResearchWorkflow** (`code/deeprag.py`) — Multi-iteration loop (default 5 iters):
  ```
  Planner → MCP Retrieval → Selector → [Browser] → Memory Update → repeat
  ```

### Agent System (`code/agent/`)
All agents use `code/api.py` for LLM calls (supports OpenAI, DeepSeek, Qwen, GLM, Gemini, Ollama via OpenAI-compatible API).

- **Planner** (`planner.py`) — Decomposes queries into subquery tree. Actions: derive (specific), expand (sibling), continue (paginate). Maintains `ResearchMemory` with experience replay.
- **Selector** (`selector.py`) — LLM-based relevance filtering per subquery. Outputs selected/discarded papers + overview.
- **Browser** (`browser.py`) — Fetches full paper text from ar5iv, extracts relevant sections via `Ar5ivParser`. Used for uncertain relevance decisions.
- **Summarizer** (`summarizer.py`) — Compresses paper abstracts. JSONL cache keyed by model name.

### Retrieval (`code/rag.py`, `code/mcp/retrieval_mcp.py`)
- `CitationRAGSystem` — Manages BM25 (rank-bm25), FAISS (faiss-gpu), and Qdrant indices.
- `retrieval_mcp.py` — Thin dispatch wrapper used by workflows to call search methods.

### Data Models (`code/structures.py`)
- `Paper`, `SubQuery`, `SubQueryState`, `ResearchMemory` — dataclasses for runtime state.
- `PlannerOutput`, `SelectorOutput` etc. — Pydantic models for structured LLM output.

### Prompts (`code/prompt.py`)
All prompt templates in one file. Multiple variants exist for planner (full_history, ablation) and selector (per browser mode).

### Metrics (`code/metrics.py`)
`MetricsCalculator` computes Recall, Precision, F1, Avg.Distance, GT Discard Rate at both retrieval and selection stages.

### Utilities (`code/utils.py`)
Mixed concerns: LLM response parsing (`parse_json_from_tag`, `remove_think_blocks`), ArXiv processing, `AgentTraceRecorder`, `CheckpointManager`.

### Build Scripts (not part of eval pipeline, live in `scripts/`)
- `scripts/build_corpus.py` — Survey paper crawler (ArXiv download, citation extraction)
- `scripts/build_benchmark.py` / `scripts/fast_build_benchmark.py` — Benchmark dataset construction
- `scripts/build_vector_db.py` — Qdrant vector DB builder
- `scripts/estimate_closed_metrics.py` / `scripts/fix_closed_metrics.py` / `scripts/plot_evaluation_summary.py` — one-off analysis tooling

Scripts live at repo root (outside `code/`) and self-configure `sys.path` to import project modules.

## Output Structure
Results go to `eval_results/{model}_{prompt}_{search}_{workflow}_{params}/`:
- `detailed_results.jsonl` — Per-query iteration metrics
- `results.json` — Aggregated evaluation
- `config.py` — Config snapshot
- Summary appended to `evaluation_summary.jsonl`

## LLM Provider Configuration
Set environment variables for the provider you need:
- `OPENAI_API_KEY` / `OPENAI_BASE_URL`
- `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL`
- `DASHSCOPE_API_KEY` / `DASHSCOPE_BASE_URL` (Qwen)
- `ZHIPU_API_KEY` / `ZHIPU_BASE_URL` (GLM)
- `GEMINI_API_KEY` / `GEMINI_BASE_URL`
- `OLLAMA_URL` (default: `http://localhost:11434`)

Provider is auto-resolved from model name prefix (e.g., `gpt-*` → OpenAI, `deepseek-*` → DeepSeek). `is_local=True` always routes to Ollama.

## Key Caveats
- No test infrastructure exists yet — changes should be verified by running `eval.py` with a short benchmark (`data/scholargym_bench_short.jsonl`).
- `config.py` uses flat module variables, not a config class. Many files do `import config` and access attributes directly.
- `utils.py` is a grab bag — check there before creating new utility functions.
- The async workflow in `deeprag.py` uses `asyncio.run()` from `eval.py`; be careful with nested event loops.
