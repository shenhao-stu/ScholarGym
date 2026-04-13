# ScholarGym Workflow Demo — API Documentation

## Overview

This is a single-page demo application built with **FastAPI** + **WebSocket**. The user submits a research query, the server simulates a multi-iteration deep research workflow, and streams real-time progress events back to the client.

- **Server**: `server.py` (FastAPI + Uvicorn)
- **Client**: `index.html` (vanilla HTML/CSS/JS, served by the server)
- **Default address**: `http://127.0.0.1:8765`

---

## HTTP Endpoints

### `GET /`

Returns the `index.html` page as an HTML response.

| Item | Value |
|------|-------|
| Method | `GET` |
| Path | `/` |
| Response | `text/html` |

---

## WebSocket Endpoint

### `ws://{host}/ws`

All research interactions happen over this persistent WebSocket connection.

---

## Client → Server Messages

The client sends JSON messages to the server. Currently only one action is supported:

### `search`

Start a research simulation.

```json
{
  "action": "search",
  "config": {
    "query": "Recent advances in retrieval-augmented generation",
    "max_iterations": 3,
    "browser_mode": "REFRESH",
    "enable_summarization": false,
    "source": "Local"
  }
}
```

#### Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | `string` | `"Recent advances in retrieval-augmented generation for scientific literature"` | The research question to search for |
| `max_iterations` | `int` | `3` | Number of research iterations (1–10) |
| `browser_mode` | `string` | `"REFRESH"` | Controls full-text fetching behavior. See [Browser Modes](#browser-modes) |
| `enable_summarization` | `bool` | `false` | Whether the Summarizer agent runs during each iteration |
| `source` | `string` | `"Local"` | Data source selection (UI-only, not consumed by server currently) |

#### Browser Modes

| Value | Description |
|-------|-------------|
| `NONE` | Retrieval only, no full-text fetching. Browser agent is skipped entirely |
| `PRE_ENRICH` | Fetch full text before the selection phase |
| `REFRESH` | Fetch full text for borderline papers after initial selection, then re-evaluate |

---

## Server → Client Events

The server streams JSON events via WebSocket. Each event has a `type` field. There are four event types:

### 1. `status`

Signals the start or end of the overall research process.

```json
{
  "type": "status",
  "status": "thinking",
  "message": "Starting deep research..."
}
```

```json
{
  "type": "status",
  "status": "done",
  "message": "Found 6 relevant papers"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | `"thinking"` \| `"done"` | Current phase |
| `message` | `string` | Human-readable status message |

---

### 2. `step`

Represents a single step in the research workflow. Steps can be nested (via `parent`) and collapsible.

```json
{
  "type": "step",
  "id": "it0-plan",
  "parent": "it0",
  "label": "Planner: 2 subqueries",
  "status": "running",
  "variant": "agent",
  "collapsible": true,
  "detail": "  ├─ [derive] \"retrieval augmented generation survey 2024\" (k=10)\n  └─ [expand] \"self-reflective RAG methods\" (k=8)\n  is_complete: false",
  "duration": 3.2
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `string` | Yes | Unique identifier for this step. Sending the same `id` again updates the existing step |
| `parent` | `string` | No | `id` of the parent step (for nesting) |
| `label` | `string` | Yes | Display text for the step |
| `status` | `"running"` \| `"done"` | Yes | Current state of this step |
| `variant` | `string` | No | Visual style hint. One of: `iteration`, `agent`, `phase`, `tool`, `info` |
| `collapsible` | `bool` | No | Whether the step can be collapsed in the UI |
| `detail` | `string` | No | Multi-line detail text (shown inside collapsible body) |
| `duration` | `number` | No | Elapsed time in seconds (shown on iteration completion) |

#### Step Hierarchy (typical)

```
Iteration (variant: iteration, collapsible)
├── Planner (variant: agent)
├── Retrieval (variant: phase, collapsible)
│   ├── BM25/Vector/Hybrid subquery (variant: tool)
│   └── ...
├── Summarizer (variant: agent) — only if enable_summarization=true
├── Selector (variant: phase, collapsible)
│   ├── Per-subquery selection (variant: agent)
│   └── ...
├── Browser (variant: agent) — only if browser_mode ≠ NONE
├── re-Selector (variant: agent) — only if browser fetched papers
└── Memory update (variant: info)
```

#### Step Update Pattern

Steps are **upserted** by `id`. A step is first sent with `status: "running"`, then later updated to `status: "done"` with a potentially different `label`. For example:

1. `{ id: "it0-plan", label: "Planner: planning subqueries...", status: "running" }`
2. `{ id: "it0-plan", label: "Planner: 2 subqueries", status: "done", detail: "..." }`

---

### 3. `metrics`

Reports evaluation metrics at the end of each iteration.

```json
{
  "type": "metrics",
  "parent": "it0",
  "metrics": {
    "recall": 0.32,
    "precision": 0.21,
    "f1": 0.25
  },
  "found_gt": 2,
  "total_gt": 4
}
```

| Field | Type | Description |
|-------|------|-------------|
| `parent` | `string` | The iteration step `id` this metric belongs to |
| `metrics.recall` | `float` | Recall score (0–1) |
| `metrics.precision` | `float` | Precision score (0–1) |
| `metrics.f1` | `float` | F1 score (0–1) |
| `found_gt` | `int` | Number of ground-truth papers found so far |
| `total_gt` | `int` | Total ground-truth papers (always 4 in this demo) |

---

### 4. `results`

Final event containing the paper results. Sent once after all iterations complete.

```json
{
  "type": "results",
  "query": "Recent advances in retrieval-augmented generation",
  "total_found": 6,
  "gt_found": 3,
  "gt_total": 4,
  "papers": [
    {
      "arxiv_id": "2401.15884",
      "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
      "authors": ["Akari Asai", "Zeqiu Wu", "..."],
      "year": 2024,
      "venue": "ICLR 2024",
      "abstract": "Despite their remarkable capabilities...",
      "citations": 487,
      "url": "https://arxiv.org/abs/2401.15884",
      "relevance": 0.95
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `query` | `string` | The original research query |
| `total_found` | `int` | Total unique papers selected |
| `gt_found` | `int` | Ground-truth papers found |
| `gt_total` | `int` | Total ground-truth papers |
| `papers` | `array` | Up to 8 papers, sorted by relevance (descending). See [Paper Object](#paper-object) |

#### Paper Object

| Field | Type | Description |
|-------|------|-------------|
| `arxiv_id` | `string` | ArXiv paper ID |
| `title` | `string` | Paper title |
| `authors` | `string[]` | List of author names |
| `year` | `int` | Publication year |
| `venue` | `string` | Conference or journal |
| `abstract` | `string` | Paper abstract |
| `citations` | `int` | Citation count |
| `url` | `string` | ArXiv URL |
| `relevance` | `float` | Relevance score (0–1) |

---

## Event Sequence Diagram

```
Client                              Server
  │                                    │
  │──── { action: "search", config } ──→│
  │                                    │
  │◄── status (thinking) ─────────────│
  │◄── step: load (running) ──────────│
  │◄── step: load children (done) ────│
  │◄── step: load (done) ─────────────│
  │                                    │
  │  ┌─── Iteration 1 ────────────────│
  │◄─│ step: it0 (running)            │
  │◄─│ step: it0-plan (running→done)  │
  │◄─│ step: it0-ret (running→done)   │
  │◄─│ step: it0-sel (running→done)   │
  │◄─│ step: it0-br (running→done)    │  ← if browser_mode ≠ NONE
  │◄─│ step: it0-mem (done)           │
  │◄─│ metrics                         │
  │◄─│ step: it0 (done)               │
  │  └────────────────────────────────│
  │                                    │
  │  ... repeat for each iteration ... │
  │                                    │
  │◄── status (done) ─────────────────│
  │◄── results ────────────────────────│
  │                                    │
```

---

## Running

```bash
pip install fastapi uvicorn
python server.py
# → Open http://127.0.0.1:8765
```
