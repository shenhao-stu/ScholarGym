"""Smoke tests for workflows.demo.DemoWorkflow.

Verifies imports, helper math, and that the workflow class can be driven
through one early-stop iteration with mocked agents (no LLM, no Qdrant).
"""
import asyncio
import json
from unittest.mock import MagicMock

import pytest


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_text(self, s):
        self.sent.append(json.loads(s))


def test_demo_workflow_imports():
    from workflows.demo import DemoWorkflow, _simple_metrics, _paper_to_payload
    assert DemoWorkflow is not None
    assert _simple_metrics is not None
    assert _paper_to_payload is not None


def test_simple_metrics_empty_gt():
    from workflows.demo import _simple_metrics
    assert _simple_metrics(set(), set()) == {
        "recall": 0.0, "precision": 0.0, "f1": 0.0,
    }
    # gt empty even with selected → all zeros
    assert _simple_metrics({"a", "b"}, set()) == {
        "recall": 0.0, "precision": 0.0, "f1": 0.0,
    }


def test_simple_metrics_partial_match():
    from workflows.demo import _simple_metrics
    m = _simple_metrics({"a", "b"}, {"a", "c"})
    assert m == {"recall": 0.5, "precision": 0.5, "f1": 0.5}


def test_simple_metrics_perfect_match():
    from workflows.demo import _simple_metrics
    m = _simple_metrics({"a", "b"}, {"a", "b"})
    assert m == {"recall": 1.0, "precision": 1.0, "f1": 1.0}


def test_paper_to_payload_minimal():
    from workflows.demo import _paper_to_payload
    from structures import Paper

    p = Paper(id="p1", title="T", abstract="A", arxiv_id="1234.5678")
    payload = _paper_to_payload(p)
    assert payload["arxiv_id"] == "1234.5678"
    assert payload["title"] == "T"
    assert payload["abstract"] == "A"
    assert payload["score"] == 0.0  # None → 0.0


def test_demo_workflow_early_stop():
    """Planner returns is_complete=True on first iter → workflow ends cleanly
    with status:done + results events, never invoking selector / browser."""
    from workflows.demo import DemoWorkflow

    ws = FakeWS()
    rag = MagicMock()
    rag.search_method = "bm25"

    wf = DemoWorkflow(rag_system=rag, ws=ws)

    # Mock planner to return is_complete=True immediately
    wf.planner = MagicMock()
    wf.planner.plan_iteration = MagicMock(
        return_value=({}, "experience", "checklist", True)
    )
    # Sentinel: if these get called we'll fail
    wf.selector = MagicMock(side_effect=AssertionError("selector should not run"))
    wf.browser = MagicMock(side_effect=AssertionError("browser should not run"))
    wf.summarizer = MagicMock(side_effect=AssertionError("summarizer should not run"))

    asyncio.run(wf.run(query="test query", max_iterations=3))

    types = [e["type"] for e in ws.sent]
    assert types[0] == "status"
    assert ws.sent[0]["status"] == "thinking"
    assert types[-1] == "results"
    assert any(e["type"] == "status" and e.get("status") == "done" for e in ws.sent)
    # results payload shape
    final = ws.sent[-1]
    assert final["query"] == "test query"
    assert final["total_found"] == 0
    assert final["papers"] == []


def test_demo_workflow_planner_failure():
    """If planner raises, workflow emits error step + done status, no crash."""
    from workflows.demo import DemoWorkflow

    ws = FakeWS()
    rag = MagicMock()
    rag.search_method = "bm25"

    wf = DemoWorkflow(rag_system=rag, ws=ws)
    wf.planner = MagicMock()
    wf.planner.plan_iteration = MagicMock(side_effect=RuntimeError("boom"))

    asyncio.run(wf.run(query="q", max_iterations=2))

    # Should have emitted an error step and a done status
    error_steps = [e for e in ws.sent
                   if e.get("type") == "step" and e.get("variant") == "error"]
    assert len(error_steps) == 1
    assert "boom" in error_steps[0]["label"]

    done_status = [e for e in ws.sent
                   if e.get("type") == "status" and e.get("status") == "done"]
    assert len(done_status) == 1
