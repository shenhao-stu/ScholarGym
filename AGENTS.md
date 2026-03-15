# AGENTS.md

## Cursor Cloud specific instructions

### Overview

ScholarGym is a Python research benchmarking tool that evaluates LLM capabilities in academic paper retrieval. The codebase lives under `code/` with data in `data/`. See `README.md` for full documentation.

### Key gotchas

- **No GPU in Cloud VM**: `requirements.txt` lists `faiss-gpu`, but the Cloud VM has no GPU. Install `faiss-cpu` instead. The update script handles this automatically.
- **Missing dependency**: `beautifulsoup4` and `httpx` are required by `code/agent/browser.py` but are not listed in `requirements.txt`. The update script installs them.
- **Python path**: All code imports assume `code/` is on `sys.path`. Run scripts from `code/` directory (e.g., `cd code && python eval.py ...`) or add `code/` to `PYTHONPATH`.
- **Large data files**: The paper corpus (`data/scholargym_paper_db.json`, ~860 MB) must be downloaded separately from HuggingFace (`shenhao/ScholarGym`) or ModelScope. The benchmark file `data/scholargym_bench.jsonl` is already in the repo.
- **BM25-only mode**: Without the paper corpus or a pre-built BM25 index (`data/bm25_index.pkl`), the eval pipeline cannot run against the full benchmark. BM25 search works without GPU and without Ollama/Qdrant.
- **LLM requirement**: The evaluation pipeline requires an LLM. By default it expects Ollama at `localhost:11434` serving `qwen3:8b`. Alternatively, set API keys (e.g., `OPENAI_API_KEY`) in `.env` for cloud LLM providers.

### Running the evaluation

```bash
cd /workspace/code
python eval.py \
    --paper_db ../data/scholargym_paper_db.json \
    --benchmark_jsonl ../data/scholargym_bench.jsonl \
    --bm25_path ../data/bm25_index.pkl \
    --workflow deep_research \
    --search_method bm25 \
    --max_iterations 5 \
    --results_per_query 10 \
    --llm_model qwen3:8b
```

### Services

| Service | Required | Notes |
|---------|----------|-------|
| Python 3.10+ | Yes | Core runtime |
| Ollama | For eval | Local LLM server at `localhost:11434` |
| Qdrant | For vector/hybrid search only | Vector DB at `localhost:6433` |

### Lint / Test / Build

This is a research codebase with no formal linter, test suite, or build step configured. Verification is done by importing modules and running the eval pipeline.
