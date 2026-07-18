# Scope of Improvement

Production-readiness analysis of `saathi-langgraph`. Issues are ranked by severity.

---

## Critical Blockers

**1. No authentication on the API**
`src/saathi/api/server.py` binds `0.0.0.0:8000` with zero auth. Any client that can reach the port gets full `run_bash`, `write_file`, and `git_commit` access on the host machine. Must be gated before any networked deployment.

**2. Shell injection on Windows**
`src/saathi/tools/shell.py` uses `subprocess.run(command, shell=True)` on Windows with a 6-entry substring denylist. The denylist is trivially bypassed (double-space, quoting). The LLM controls `command` directly.

**3. `block_paths` hook does not cover `run_bash`**
`src/saathi/hooks/runner.py` only intercepts `write_file` and `patch_file`. The shell tool can write `.env`, private keys, etc., with no hook blocking it -- defeating the entire `block_paths` safety model.

---

## High Severity

**4. Session deserialization loses `tool_calls` / `tool_call_id`**
`src/saathi/session/manager.py` (line ~68) serializes only `type` and `content`. Restored sessions with tool-call history feed broken messages back into the graph, which can crash the next LLM call.

**5. `_brave_search` blocks the event loop**
`src/saathi/tools/search.py` (line ~96) calls `httpx.get(...)` (synchronous) inside an async context. Any Brave search stalls the entire event loop. Needs `httpx.AsyncClient`.

**6. Memory tools use a module-level singleton, bypassing DI**
`src/saathi/tools/memory_tools.py` creates its own `_store = MemoryStore()` at import time. The injected store in `build_graph` and this singleton both hit the same JSON files but are independent objects -- causing test isolation failures and an architecturally fragile dual-singleton pattern.

---

## Medium Severity

| # | File | Issue |
| --- | ------ | ------- |
| 7 | `src/saathi/api/routes/chat.py` | SSE `event_generator` has no error handling -- mid-stream exceptions silently close the stream with no error event sent to the client |
| 8 | `src/saathi/api/routes/sessions.py` | `GET /sessions/{id}/history` returns 404 for a brand-new session instead of an empty-history 200 |
| 9 | `src/saathi/ui/commands.py` (line ~113) | `/rollback` updates the graph checkpoint but not the in-memory `messages` list in `cli.py` -- causes divergence on the next turn |
| 10 | `src/saathi/memory/store.py` | No file locking -- concurrent API requests can corrupt the JSON memory files |
| 11 | `src/saathi/tools/search.py` (line ~49) | `file_glob` is passed directly to `Path.glob` -- a glob like `../../**/*` walks outside the target directory |
| 12 | (missing) | No `Dockerfile` or container config for the API server |
| 13 | `.github/workflows/ci.yml` | CI only tests Linux (ubuntu-latest) despite explicit Windows support code paths in `shell.py` and `cli.py` |

---

## Low / Minor

- `src/saathi/agent/nodes.py` (line ~26) -- `SAATHI.md` loaded once at graph-build time; mid-session edits require a `/model` switch to take effect
- `src/saathi/tools/git.py` (line ~67) -- `git_log` silently caps at 50 commits regardless of user input
- `src/saathi/review.py` (line ~22) -- imports `_git` (underscore-prefixed internal) across module boundaries; fragile API boundary
- `src/saathi/api/static/index.html` -- hand-rolled regex markdown parser; session list is in-memory only and lost on page refresh

---

## What Is Already Production-Grade

- Parallel tool execution with semaphore + guaranteed `ToolMessage` per `tool_call_id`
- Retry logic scoped correctly to connection-establishment errors only (not read timeouts)
- History compaction that never orphans a `ToolMessage`
- Pydantic-settings config with `.env` support and no hardcoded secrets
- 114 offline tests, most with `tmp_path` isolation
- Regression test guarding the ReAct loop wiring (`test_agent_only_ends_conditionally`)
- Hooks system as clean middleware (pre_tool, post_tool, post_turn, block_paths)
- Graceful MCP failure -- broken servers never block startup
- `ARCHITECTURE.md` with documented trade-offs and Mermaid diagrams

---

## Recommended Fix Order

1. API authentication (Critical #1)
2. Shell injection on Windows (Critical #2)
3. `block_paths` coverage for `run_bash` (Critical #3)
4. Session serialization of `tool_calls` (High #4)
5. Sync HTTP in async context -- Brave search (High #5)
6. Memory DI singleton (High #6)
