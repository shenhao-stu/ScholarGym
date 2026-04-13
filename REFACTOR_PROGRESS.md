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
| 1.4 | Group per-model configs under configs/ | | | | |
| 1.5 | Move simplerag.py / deeprag.py → workflows/ | | | | |
| 1.6 | Move test_qwen.py → tests/ | | | | |
| 1.7 | Section config.py with header comments | | | | |
| verify | Final green + tag v0.2.0-phase1-done | | | | |

## Phase 2 — Demo integration

| # | Sub-phase | Commit | Notes |
|---|---|---|---|
| 2.1 | Write workflows/demo.py skeleton | | |
| 2.2 | Rewrite server.py + split server_mock.py | | |
| 2.3 | Align index.html with real workflow shapes | | |
| 2.4 | Add DemoWorkflow smoke tests | | |
| 2.5 | Manual end-to-end smoke + tag v0.2.0 | | |

---

## Incidents / decisions

- **1.0** `api.py` has `skip-worktree` flag (keeps local API keys out of HEAD). Committed only the cassette additions (120 lines) rebuilt on HEAD base, then restored working copy with retry+keys+cassette, re-locked skip-worktree. Zero key leakage.
- **1.1** `scripts/` was in `.gitignore` — removed that line. 3 pre-existing untracked files in `scripts/` (dedup_qdrant.py, start_services.sh, test_qdrant_filter.py) intentionally NOT committed per user instruction.
- **1.3** Package was initially created as `prompts/` (plural); renamed to `prompt/` (singular) to match the old module name so zero existing import statements needed editing.
