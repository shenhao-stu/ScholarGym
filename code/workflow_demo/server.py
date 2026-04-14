"""
ScholarGym demo — REAL pipeline bridge.

WebSocket server that drives workflows.demo.DemoWorkflow against the real
Planner / Selector / Browser / Summarizer + CitationRAGSystem stack. Requires
Qdrant + Ollama (or whatever LLM provider config.py points at) to be running.

For a zero-dependency mock (UI development), use server_mock.py instead.

Run:  python code/workflow_demo/server.py   (port 8765)
"""
import os
import sys
import json
from pathlib import Path

os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

# Make `code/` importable regardless of cwd.
_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import config
from rag import CitationRAGSystem
from workflows.demo import DemoWorkflow
from logger import get_logger

logger = get_logger(__name__)

app = FastAPI()

# Shared RAG system loaded once at startup.
rag_system: CitationRAGSystem | None = None


@app.on_event("startup")
async def _startup():
    global rag_system
    logger.info("[demo] loading CitationRAGSystem...")
    rag_system = CitationRAGSystem(search_method=config.DEFAULT_SEARCH_METHOD)
    rag_system.load_or_build_indices(
        paper_db_path=config.PAPER_DB_PATH,
        bm25_path=config.BM25_PATH,
        rebuild=False,
    )
    logger.info("[demo] RAG system ready (search_method=%s)",
                rag_system.search_method)


@app.get("/")
async def get():
    return HTMLResponse(
        (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
    )


def _lookup_gt(qid: str | None) -> list[str] | None:
    """Look up ground-truth arxiv_ids for a benchmark qid, if given."""
    if not qid:
        return None
    try:
        with open(config.BENCHMARK_PATH) as f:
            for line in f:
                q = json.loads(line)
                if q.get("qid") == qid:
                    return [cp.get("arxiv_id") for cp in q.get("cited_paper", [])
                            if cp.get("arxiv_id")]
    except FileNotFoundError:
        logger.warning("[demo] benchmark file not found: %s", config.BENCHMARK_PATH)
    except Exception as e:
        logger.warning("[demo] gt lookup failed: %s", e)
    return None


@app.websocket("/ws")
async def ws_ep(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                msg = {"action": data}

            if msg.get("action") != "search":
                continue

            cfg = msg.get("config", {})

            # search_method is locked at startup (switching index requires
            # reloading BM25 or Qdrant). We log but don't honor per-request
            # overrides, so the UI should surface the active method.
            requested = cfg.get("search_method")
            if requested and rag_system and requested != rag_system.search_method:
                logger.info(
                    "[demo] search_method override %r ignored (active=%r)",
                    requested, rag_system.search_method,
                )

            workflow = DemoWorkflow(rag_system=rag_system, ws=ws)
            try:
                await workflow.run(
                    query=cfg.get("query", ""),
                    max_iterations=int(cfg.get("max_iterations", 3)),
                    browser_mode=cfg.get("browser_mode", "NONE"),
                    enable_summarization=bool(cfg.get("enable_summarization", False)),
                    gt_arxiv_ids=_lookup_gt(cfg.get("benchmark_qid")),
                )
            except Exception as e:
                logger.exception("[demo] workflow crashed")
                try:
                    await ws.send_text(json.dumps({
                        "type": "status",
                        "status": "done",
                        "message": f"Workflow error: {e}",
                    }))
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
