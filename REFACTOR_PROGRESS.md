# Refactor Progress — patch branch

**Plan:** `/data/yanghang/.claude/plans/refactored-whistling-mountain.md`
**Starting point:** `v0.1.0` tag = `79a96c0`
**Baseline pytest:** 195 passed
**Behavior check:** LLM cassette replay + `scripts/diff_snapshot.py` (strict fields only)

---

## Phase 1 — File reorganization

| # | Sub-phase | Commit | pytest | cassette replay | Notes |
|---|---|---|---|---|---|
| 1.0 | Freeze v0.1.0 baseline + add LLM cassette infra | `b7d1752` | 195 pass / 1 skip | — (baseline capture) | 9 cassette files, 88KB |
| 1.1 | Move one-off scripts out of code/ | `c48be00` | 195 pass / 1 skip | — | 7 scripts → scripts/, conftest adds scripts/ to sys.path |
| 1.2 | Split utils.py → utils/ package | `24d0024` | 195 pass / 1 skip | ✅ strict fields match | 4 submodules (llm_parsing / metrics_helpers / trace / checkpoint) |
| 1.3 | Split prompt.py → prompt/ package | `91c2986` | 195 pass / 1 skip | ✅ strict fields match | 6 submodules. Package named 'prompt' (singular) so all `from prompt import` call sites untouched. Byte-identical prompts verified via cassette key survival. |
| 1.4 | Group per-model configs under configs/ | `70d44fd` | 195 pass / 1 skip | ✅ | 4 files moved to code/configs/ |
| 1.5 | Move simplerag.py / deeprag.py → workflows/ | `384927d` | 195 pass / 1 skip | ✅ | + workflows/__init__.py re-exports; mock patch paths updated |
| 1.6 | Move test_qwen.py → tests/ | `fdfdc5c` | 195 pass / 1 skip | — | renamed to test_smoke_qwen.py |
| 1.7 | Section config.py with header comments | `5aabe60` | 195 pass / 1 skip | — | banner-style headers, zero variable changes |
| verify | Final green + tag v0.2.0-phase1-done | (tag) | 195 pass | ✅ | code/ top-level down to 7 .py files |

## Phase 2 — Demo integration

| # | Sub-phase | Commit | Notes |
|---|---|---|---|
| 2.1 | Write workflows/demo.py skeleton | `ad81b30` | ~380 lines async-native; reuses Planner/Selector/Browser/PaperSummarizer + mcp.retrieval_mcp.search_papers. Corrected vs plan: Browser.browse_papers (not browse), PaperSummarizer.batch_summarize (not summarize_batch), Selector needs user_query + planner_checklist as separate args, Planner.plan_iteration is sync and returns Dict[int, SubQuery]. |
| 2.2 | Rewrite server.py + split server_mock.py | `095705f` | server.py is ~120-line FastAPI bridge (loads RAG once, drives DemoWorkflow per request, optional GT lookup from benchmark_qid). server_mock.py is the old simulate-based mock on port 8766. |
| 2.3 | Align index.html with real workflow shapes | `1004f4b` | renderResults tolerates missing authors/year/venue/citations/url/relevance; falls back to score, synthesizes arxiv link from arxiv_id, hides "Ground truth: 0/0" for non-benchmark queries. re-Selector / Hybrid hardcoding turned out to already be UI-agnostic (label rendering follows server-sent strings). |
| 2.4 | Add DemoWorkflow smoke tests | `f23cec4` | 7 tests: imports, _simple_metrics math (3 cases), _paper_to_payload shape, early-stop run with mocked planner, planner-failure error path. 202 pass total. |
| 2.5 | Manual end-to-end smoke + tag v0.2.0 | `cbdf27c` (+ tag) | Real ws client → server.py: planner→3 subqueries→BM25 retrieval(30 papers)→selector(11 selected)→results event w/ real arxiv_ids. Mock server also verified. Bugfix: starlette raises RuntimeError (not WebSocketDisconnect) when peer drops mid-loop; both now caught. |

---

## Incidents / decisions

- **1.0** `api.py` has `skip-worktree` flag (keeps local API keys out of HEAD). Committed only the cassette additions (120 lines) rebuilt on HEAD base, then restored working copy with retry+keys+cassette, re-locked skip-worktree. Zero key leakage.
- **1.1** `scripts/` was in `.gitignore` — removed that line. 3 pre-existing untracked files in `scripts/` (dedup_qdrant.py, start_services.sh, test_qdrant_filter.py) intentionally NOT committed per user instruction.
- **1.3** Package was initially created as `prompts/` (plural); renamed to `prompt/` (singular) to match the old module name so zero existing import statements needed editing.
- **2.1** Plan skeleton had several wrong agent method names. Verified actual signatures from production code before writing demo.py: `Browser.browse_papers` (not `browse`), `PaperSummarizer.batch_summarize` (not `summarize_batch`), `Selector.decide_for_subquery` takes `user_query: str` + `planner_checklist: str` as separate args, `Planner.plan_iteration` is sync and returns `Dict[int, SubQuery]`. Also: `CitationRAGSystem` has no `search_citations` method — must dispatch via `mcp.retrieval_mcp.search_papers` and convert dict results to `Paper` dataclasses.
- **2.5** Real-pipeline e2e ran cleanly: 1 iteration, 3 subqueries, 30 BM25 hits, 11 selected, 8 returned in results payload. First test attempt with `max_iterations=2` actually completed on server side (saw 21 selections in log) but exceeded client 180s timeout — pipeline is just slow with real LLM (~3 min for 2 iters), not a bug. Discovered minor bugfix: starlette raises `RuntimeError("WebSocket is not connected")` instead of `WebSocketDisconnect` when peer drops mid-loop. Both now caught.
