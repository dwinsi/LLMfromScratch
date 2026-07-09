# Building Production AI Agents: A Complete Engineering Guide

A book-style reference using **saathi-langgraph** — a local coding agent powered by LangGraph and Ollama — as the worked example throughout. Every concept is grounded in real, running code from the project.

---

## Who this is for

Experienced Python engineers who want to build production-grade LLM agent systems. Not a quick-start — a deep engineering reference.

---

## Chapters

### Part I — Foundations

| Chapter | Title | Key topics |
| --------- | ------- | ------------ |
| [00](00-preface.md) | Preface | What we build, why local LLMs, how to read this book |
| [01](01-python-foundations.md) | Python Foundations | asyncio, gather, Semaphore, TypedDict, Pydantic v2, PEP 695, structlog, pytest-asyncio |
| [02](02-llm-ecosystem.md) | The LLM Ecosystem | Transformers, tokenization, context windows, ReAct, structured outputs, model families |
| [03](03-ollama.md) | Ollama | Local LLM serving, HTTP API, ChatOllama, tool calling, performance tuning, remote setup |

### Part II — LangGraph

| Chapter | Title | Key topics |
| --------- | ------- | ------------ |
| [04](04-langchain-foundations.md) | LangChain Foundations | Message hierarchy, Runnable protocol, `@tool`, `bind_tools`, `astream_events` |
| [05](05-langgraph-core.md) | LangGraph Core | StateGraph, nodes, edges, `tools_condition`, reducers, checkpointing, the ReAct loop |
| [06](06-agent-state-and-reducers.md) | Agent State & Reducers | TypedDict, `Annotated`, `add_messages`, custom reducers, `aupdate_state`, rollback |
| [07](07-tools-and-tool-node.md) | Tools & the Tool Node | All 15 tools, parallel execution, semaphore, hooks integration, MCP normalization |
| [08](08-checkpointing-and-rollback.md) | Checkpointing & Rollback | `AsyncSqliteSaver`, thread model, `aget_state_history`, `/rollback`, `aiosqlite` |
| [09](09-streaming.md) | Streaming | `astream_events`, token chunks, tool call streaming, Rich console, `--print` mode |

### Part III — Production Features

| Chapter | Title | Key topics |
| --------- | ------- | ------------ |
| [10](10-memory-systems.md) | Memory Systems | Two-scope JSON store, SAATHI.md, `/init`, `/revise-saathi-md`, semantic memory |
| [11](11-hooks-and-security.md) | Hooks & Security | `block_paths`, pre/post tool hooks, denylist, threat model, defense in depth |
| [12](12-history-compaction.md) | History Compaction | Token budgeting, `split_for_compaction`, LLM summarization, fresh thread strategy |
| [13](13-mcp-protocol.md) | MCP Protocol | MCP transports, `MultiServerMCPClient`, `_result_to_text`, echo server, ecosystem |
| [14](14-code-review-workflow.md) | Code Review Workflow | Multi-reviewer concurrency, `Finding` model, tolerant JSON parsing, Rich display |
| [15](15-testing-llm-apps.md) | Testing LLM Apps | Fake LLMs, `asyncio_mode=auto`, regression guards, `pytest.mark.live`, coverage |

### Part IV — Engineering Excellence

| Chapter | Title | Key topics |
| --------- | ------- | ------------ |
| [16](16-configuration-12factor.md) | Configuration & 12-Factor | `pydantic-settings`, `SAATHI_` prefix, `.env`, computed fields, testing config |
| [17](17-cli-design.md) | CLI Design | Typer, REPL loop, slash commands, `--print` mode, token footer, Windows Unicode |
| [18](18-langsmith-observability.md) | LangSmith Observability | Tracing, debugging, datasets, evals, `@traceable`, privacy, alternatives |
| [19](19-production-patterns.md) | Production Patterns | CI/CD, Docker, FastAPI, rate limiting, multi-tenancy, metrics, Ollama scaling |
| [20](20-future-of-llms.md) | Future of LLMs | Reasoning models, MCP ecosystem, multi-agent systems, RAG, evals, open problems |

---

## Quick reference

```folder
saathi-langgraph/
├── src/saathi/
│   ├── agent/
│   │   ├── graph.py        ← Chapter 5, 8
│   │   ├── nodes.py        ← Chapter 5, 6
│   │   ├── state.py        ← Chapter 6
│   │   ├── tool_node.py    ← Chapter 7
│   │   └── prompts.py      ← Chapter 2, 10
│   ├── tools/              ← Chapter 7
│   ├── hooks/runner.py     ← Chapter 11
│   ├── memory/store.py     ← Chapter 10
│   ├── compaction.py       ← Chapter 12
│   ├── mcp_client.py       ← Chapter 13
│   ├── review.py           ← Chapter 14
│   ├── retry.py            ← Chapter 1, 3
│   ├── config.py           ← Chapter 16
│   ├── cli.py              ← Chapter 17
│   └── logging_config.py   ← Chapter 1, 18
└── tests/                  ← Chapter 15
```

---

## Approximate page count

| Chapter | Lines | Est. pages |
| --------- | ------- | ------------ |
| 00 Preface | ~650 | ~26 |
| 01 Python Foundations | ~1,500 | ~60 |
| 02 LLM Ecosystem | ~1,601 | ~64 |
| 03 Ollama | ~2,180 | ~87 |
| 04 LangChain Foundations | ~1,079 | ~43 |
| 05 LangGraph Core | ~1,340 | ~54 |
| 06 Agent State & Reducers | ~1,600 | ~64 |
| 07 Tools & Tool Node | ~1,800 | ~72 |
| 08 Checkpointing | ~1,500 | ~60 |
| 09 Streaming | ~1,970 | ~79 |
| 10 Memory Systems | ~2,183 | ~87 |
| 11 Hooks & Security | ~2,105 | ~84 |
| 12 History Compaction | ~1,367 | ~55 |
| 13 MCP Protocol | ~1,487 | ~59 |
| 14 Code Review Workflow | ~1,225 | ~49 |
| 15 Testing LLM Apps | ~1,680 | ~67 |
| 16 Configuration | ~1,000 | ~40 |
| 17 CLI Design | ~1,400 | ~56 |
| 18 LangSmith | ~1,200 | ~48 |
| 19 Production Patterns | ~1,966 | ~79 |
| 20 Future of LLMs | ~1,220 | ~49 |
| **Total** | **~31,053** | **~1,242** |
