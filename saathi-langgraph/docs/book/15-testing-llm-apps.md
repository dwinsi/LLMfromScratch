# Chapter 15 — Testing LLM Applications: Strategies and Patterns

> *"The hardest part of testing LLM apps is accepting what you cannot test directly and building a suite that tests everything else thoroughly."*

---

## Table of Contents

1. [Why Testing LLM Apps is Hard](#1-why-testing-llm-apps-is-hard)
2. [The Testing Pyramid for LLM Apps](#2-the-testing-pyramid-for-llm-apps)
3. [`pytest-asyncio` and `asyncio_mode = "auto"`](#3-pytest-asyncio-and-asyncio_mode--auto)
4. [`tmp_path` — Isolated File System](#4-tmp_path--isolated-file-system)
5. [Fake LLMs — The Core Pattern](#5-fake-llms--the-core-pattern)
6. [Fake LLMs with Tool Calls](#6-fake-llms-with-tool-calls)
7. [`structlog.testing.capture_logs`](#7-structlogtestingcapture_logs)
8. [Testing the Graph](#8-testing-the-graph)
9. [The Regression Guard Pattern](#9-the-regression-guard-pattern)
10. [`monkeypatch` for Environment Variables](#10-monkeypatch-for-environment-variables)
11. [Testing Parallel Execution](#11-testing-parallel-execution)
12. [Testing Hooks](#12-testing-hooks)
13. [`pytest.mark.live` — Opt-in E2E Tests](#13-pytestmarklive--opt-in-e2e-tests)
14. [The Ollama Readiness Check](#14-the-ollama-readiness-check)
15. [Testing MCP with a Real Server](#15-testing-mcp-with-a-real-server)
16. [CI Configuration](#16-ci-configuration)
17. [Coverage](#17-coverage)
18. [Property-Based Testing with Hypothesis](#18-property-based-testing-with-hypothesis)
19. [Test Organization](#19-test-organization)
20. [The "Test Doubles" Vocabulary](#20-the-test-doubles-vocabulary)

---

## 1. Why Testing LLM Apps is Hard

Testing a traditional web API or a data pipeline is difficult but tractable. The components are deterministic: given the same input, you get the same output. You can write assertions that are precise and reliable. A test that passes today will pass tomorrow.

Testing an LLM application breaks this assumption at its foundation.

### The Non-Determinism Problem

The most visible challenge is that LLMs are non-deterministic. Set `temperature=0` and you get nearly-deterministic output from most models — but not exactly deterministic. Quantized models, batching behavior, and floating-point accumulation differences across hardware all mean that the same prompt can produce slightly different tokens on the same machine on different runs.

At higher temperatures (saathi uses `temperature=0.1`), the non-determinism is intentional and more significant. Two runs of the same prompt may produce responses that are semantically equivalent but textually different. A test that says `assert response == "I found 3 issues"` is fragile.

More fundamentally: you cannot write a unit test that asserts "the LLM gave the right answer." The LLM's answer depends on the model weights, the hardware, the sampling parameters, and the phase of the moon. Trying to freeze this in a test produces tests that fail intermittently — the worst possible outcome for a test suite.

### The Speed Problem

LLM calls are slow. A single turn on a 12B parameter model running locally on a mid-range GPU takes 3–15 seconds, depending on context length and output length. A suite of 113 tests that each makes a real LLM call would take 5–30 minutes to run. That is not a test suite — it is a batch job.

Fast feedback loops are one of the core values of good testing. If the test suite takes 20 minutes to run, developers stop running it. Tests that are not run do not prevent bugs.

### The Cost Problem

Even if speed were not a concern, real LLM calls in tests are expensive when running against commercial APIs. At a few cents per 1,000 tokens, a 113-test suite that each passes a 500-token prompt and receives a 500-token response would cost a few dollars per run. At 50 CI runs per day (developers pushing, PRs being merged, nightly runs), that is hundreds of dollars per month for the test suite alone.

Saathi targets Ollama (local LLMs), so the marginal cost is compute, not dollars. But the speed problem remains, and the principle applies directly to cloud-hosted LLM development.

### The Semantic Gap Problem

Even if you accept the cost and slowness of real LLM calls in tests, you face a deeper problem: what do you assert? The LLM's output is a natural language string. You cannot assert exact equality. You can assert containment ("the response contains 'PONG'"), but this is fragile in different ways — a brief model update might produce "Pong!" instead of "PONG."

More importantly, the valuable behaviors you want to test — "does the agent use the right tool in this situation?", "does the graph route correctly after a tool call?", "does compaction preserve recent messages?" — are not in the LLM's output. They are in the system's behavior around the LLM.

### The Solution: Test Everything Around the LLM

The resolution to all of these challenges is to recognize what you actually need to test:

1. **The scaffolding:** Does the graph have the right structure? Do nodes connect correctly? Does the checkpointer persist state?
2. **The data processing:** Does compaction correctly split and summarize? Does the review parser extract JSON from various response shapes?
3. **The infrastructure:** Do hooks block the right paths? Do tools read and write files correctly? Does config parse environment variables?
4. **The LLM interface:** Does the code call the LLM with the right messages? Does it handle LLM errors gracefully?

None of these require a real LLM. A fake LLM that returns a predetermined response tests all of the code around the LLM call without touching Ollama.

The LLM itself — the question "does the model give sensible answers?" — is tested by end-to-end tests that are opt-in, run rarely, and have generous assertions.

---

## 2. The Testing Pyramid for LLM Apps

The classic testing pyramid says: many unit tests, fewer integration tests, very few end-to-end tests. The reasoning is that unit tests are fast, cheap, and precise; end-to-end tests are slow, expensive, and fragile.

For LLM applications, this structure is even more important, because the LLM component is the slowest, most expensive, and most fragile part of the system.

### Level 1: Unit Tests with Fake LLMs (Offline)

These tests use fake LLMs that return predetermined responses. They do not touch Ollama, the network, or the filesystem (except through `tmp_path`). They run in milliseconds.

These tests cover:

- Data model validators (confidence clamping, severity normalization)
- Parsing functions (JSON extraction, findings parsing)
- Tool implementations (file reading, shell execution)
- Configuration parsing
- Graph structure (does the compiled graph have the right nodes and edges?)
- Hook behavior (does the block list work? do pre-tool hooks run?)
- Compaction logic (splitting, summarizing, token estimation)
- Logging output

Level 1 tests are the core of saathi's suite. They should be fast enough to run on every file save.

### Level 2: Integration Tests with Real Components (Offline)

These tests use real implementations of multiple components working together, but still no LLM calls. They test more complex interactions.

Examples:

- The hooked tool node handling a mixed batch of blocked, allowed, and unknown tool calls
- The parallel execution test: 5 real async tools running concurrently
- MCP config loading with various JSON shapes
- Graph compilation with a SQLite checkpointer

These tests may be slightly slower (a few hundred milliseconds) due to real async operations and filesystem I/O, but they still run offline.

### Level 3: E2E Tests with Real Ollama (Live, Opt-in)

These tests call a real running Ollama with a real model. They are marked `@pytest.mark.live` and excluded from normal CI runs. They run only when explicitly invoked with `pytest -m live` on a machine with Ollama running.

The single e2e test in saathi (`test_e2e_live.py`) sends "Reply with exactly the word: PONG" to the full graph and asserts that the agent produced a non-empty response. It does not assert the exact content — that would be too fragile.

### Saathi's Distribution

Saathi's test suite at the time of writing:

- **~113 offline tests** across 20 test files, running in roughly 2–5 seconds
- **1 live e2e test** that requires Ollama, excluded by default

This distribution is intentional and worth preserving. The test suite should remain fast enough that developers run it before every push without thinking about it.

---

## 3. `pytest-asyncio` and `asyncio_mode = "auto"`

Saathi's codebase is heavily async. LangGraph is async. LangChain's `ainvoke` is async. The tool node is async. The CLI loop is async. Any test that exercises these components must be an async function.

By default, pytest does not run async test functions. A plain `async def test_foo() -> None:` is silently treated as a sync function, which means it is collected but its body never actually runs — a dangerous silent pass.

### The `pytest-asyncio` Plugin

`pytest-asyncio` extends pytest to understand async test functions. It installs an asyncio event loop and runs async tests inside it.

The configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-ra -q -m 'not live'"
markers = [
    "live: end-to-end test that requires a running Ollama (run with `pytest -m live`)",
]
```

The critical line is `asyncio_mode = "auto"`. This tells `pytest-asyncio` to automatically detect async test functions and run them in an event loop, without requiring any decorator.

### What `asyncio_mode = "auto"` Means in Practice

Without `asyncio_mode = "auto"`:

```python
import pytest

@pytest.mark.asyncio  # required decorator
async def test_something() -> None:
    ...
```

With `asyncio_mode = "auto"`:

```python
async def test_something() -> None:  # no decorator needed
    ...
```

Every async function whose name starts with `test_` is automatically recognized and run as an async test. This eliminates 40+ decorator lines across saathi's test suite and reduces the chance of accidentally writing a sync wrapper around an async body.

### Async Tests Look Like Sync Tests

With `asyncio_mode = "auto"`, async tests are written exactly like sync tests:

```python
# Sync test
def test_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.max_parallel_tools == 8


# Async test — same structure, just `async def`
async def test_run_review_filters_below_confidence() -> None:
    llm = FakeLLM(_resp([{"severity": "high", "confidence": 90, "issue": "bug"}]))
    findings = await run_review(llm, "diff", reviewers={"solo": "look"}, min_confidence=70)
    assert len(findings) == 1
```

The only difference is `async def` and `await`. No imports of asyncio, no event loop management, no special test helpers. pytest-asyncio handles all of it.

### Async Fixtures

`asyncio_mode = "auto"` also makes async fixtures work without decoration:

```python
@pytest.fixture
async def connected_client() -> AsyncIterator[MyClient]:
    client = await MyClient.connect("localhost:8080")
    yield client
    await client.close()
```

This fixture is automatically recognized as async and awaited when injected into tests.

### Pitfall: Mixing Sync and Async

One pitfall: an `async def` fixture injected into a `def` (sync) test will not work — the fixture's coroutine is returned, not the result. If you see a test that receives a coroutine object instead of the fixture value, you have a sync test using an async fixture. The fix is to either make the test async or make the fixture sync.

---

## 4. `tmp_path` — Isolated File System

Many tests need to read from and write to files. Using real paths (the project directory, `C:\Users\...`, `/tmp`) creates two problems:

1. **Non-isolation.** If a test writes `./output.json`, it writes to the project directory. Another test might read it unexpectedly. Tests interfere with each other.
2. **Cleanup.** Tests that write real files can leave artifacts. If a test fails midway, it may leave partial files.

`tmp_path` is pytest's built-in fixture for isolated file system access. It creates a unique temporary directory per test function and cleans it up after the test.

### Using `tmp_path`

```python
from pathlib import Path


def test_writes_correct_file(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    write_result(output)
    assert output.read_text(encoding="utf-8") == '{"ok": true}'
```

The `tmp_path` fixture is injected by name. pytest creates a directory like `/tmp/pytest-of-user/pytest-123/test_writes_correct_file0/` and assigns it to `tmp_path`. The test can create files, subdirectories, and anything else in this directory without affecting other tests.

### `tmp_path` in `conftest.py`

Saathi's `conftest.py` uses `tmp_path` in two fixtures:

```python
@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """A small readable file with known content, isolated in tmp_path."""
    p = tmp_path / "sample.txt"
    p.write_text("SAMPLE_CONTENT_123", encoding="utf-8")
    return p


@pytest.fixture
def isolated_memory(tmp_path: Path) -> MemoryStore:
    """A MemoryStore whose global/project files live under tmp_path (no real writes)."""
    store = MemoryStore()
    store._global_path = tmp_path / "global" / "memory.json"
    store._project_path = tmp_path / "project" / "memory.json"
    return store
```

`sample_file` creates a known-content file and returns its path. Tests that need to read a file use this fixture instead of creating the file inline. This makes the setup reusable across the test suite.

`isolated_memory` creates a `MemoryStore` but redirects its file paths into `tmp_path`. The `MemoryStore` will read and write JSON files, but it will do so in the temp directory, not in the real global or project directories. A test using `isolated_memory` can write memories without affecting the developer's real memory files.

### Using `tmp_path` in Tool Tests

From `test_tools.py`:

```python
async def test_write_and_read_file(tmp_path: Path) -> None:
    path = str(tmp_path / "hello.txt")
    result = await write_file.ainvoke({"path": path, "content": "hello"})
    assert "wrote" in result.lower()
    result2 = await read_file.ainvoke({"path": path})
    assert result2 == "hello"
```

The `write_file` and `read_file` tools are real implementations. `tmp_path` ensures the test writes to an isolated directory, and the cleanup is automatic. No `try/finally` to clean up the file. No risk of leaving artifacts.

### `tmp_path_factory` for Session-Scoped Fixtures

For fixtures shared across an entire test session (not per-test), use `tmp_path_factory`:

```python
@pytest.fixture(scope="session")
def shared_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("db")
    return base / "session.db"
```

This creates one temp directory for the entire session, shared across tests that use the fixture.

---

## 5. Fake LLMs — The Core Pattern

The central technique for testing LLM applications offline is the fake LLM. A fake LLM implements the same interface as a real LLM — specifically, the `ainvoke` method — but returns a predetermined response instead of calling the model.

### The Simplest Fake

From `test_review.py`:

```python
class FakeLLM:
    """Returns a fixed response for every reviewer call."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        return AIMessage(content=self.response)
```

This is the minimal fake LLM. It:

- Accepts `ainvoke(messages, config=None)` — the same signature as a real LangChain model
- Returns an `AIMessage` with the predetermined text content
- Counts invocations (useful for asserting concurrency and call counts)
- Never raises, never touches the network, completes in microseconds

Any code that calls `llm.ainvoke([...])` and uses the result's `.content` attribute will work with `FakeLLM` as a drop-in for `ChatOllama`.

### The Compaction Fake

From `test_compaction.py`, a slightly more sophisticated fake:

```python
class FakeLLM:
    """Minimal async LLM stub that returns a fixed summary."""

    def __init__(self, summary: str = "THE SUMMARY") -> None:
        self.summary = summary
        self.calls: list[list[BaseMessage]] = []

    async def ainvoke(self, messages, config=None):
        self.calls.append(messages)
        return AIMessage(content=self.summary)
```

This version records the full message list passed to each call (`self.calls: list[list[BaseMessage]]`). This enables assertions on what messages were sent to the LLM:

```python
assert llm.calls, "summarizer should have been invoked once"
```

Recording the call arguments turns the fake from a stub (just returns a canned response) into a spy (records what it was called with). This is useful when you want to verify not just that the LLM was called, but what it was called with.

### Why Not Use `FakeListChatModel`?

LangChain provides `FakeListChatModel` in `langchain_core.language_models.fake`. It takes a list of responses and returns them in order. It is a production-quality fake with full `BaseChatModel` support.

Saathi's tests use handwritten fakes instead. The reason is simplicity: `FakeListChatModel` has a complex initialization (it is a full `BaseChatModel` subclass with all the machinery), requires importing from a potentially unstable internal module, and adds a dependency that may change across LangChain versions.

The handwritten fake is 10 lines. It is stable. It is easy to modify. It is easy to understand. For the level of testing saathi needs, it is sufficient.

That said, `FakeListChatModel` is the right choice when you need:

- A fake that models multiple sequential calls returning different responses
- A fake that participates in LangChain's tracing and callback system
- A fake that needs to be compatible with LangChain's full `BaseChatModel` interface

For basic unit testing — "call the LLM, get a response, process the response" — the handwritten fake is preferable.

### The Duck-Typing Advantage

Python's duck typing means that a fake LLM does not need to inherit from any base class. It just needs to implement the method that the code under test calls. In saathi's case, that is `async def ainvoke(self, messages, config=None)`.

The type annotation `LanguageModelLike` in the function signatures is a structural type (or a Protocol), not a concrete class. Any object that has an `ainvoke` method of the right signature satisfies it at runtime.

This makes fake LLMs trivial to write: no inheritance, no metaclass magic, no registration. Just implement the method.

---

## 6. Fake LLMs with Tool Calls

Testing the graph's tool routing requires a fake LLM that does not just return text — it returns structured tool calls. An LLM in the ReAct loop says "call `read_file` with `path='src/main.py'`" by returning an `AIMessage` with `tool_calls` populated.

### The `ai_with_tool_calls` Helper

From `tests/helpers.py`:

```python
from langchain_core.messages import AIMessage


def ai_with_tool_calls(calls: list[dict]) -> dict:
    """Build a graph state whose last message carries the given tool calls."""
    return {
        "messages": [AIMessage(content="", tool_calls=calls)],
        "context_paths": [],
        "mode": "default",
        "session_id": "test",
    }


def tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}
```

`tool_call()` builds a tool call dict in the format LangChain expects. The `"type": "tool_call"` field is required by LangChain's message handling.

`ai_with_tool_calls()` builds a complete graph state dict with an `AIMessage` containing the given tool calls. This state is passed directly to a node (like the tool node) in tests.

### Using the Helpers

From `test_parallel.py`:

```python
async def test_calls_run_in_parallel() -> None:
    node = make_hooked_tool_node([slow_echo], _no_hooks())
    calls = [tool_call("slow_echo", {"label": str(i)}, f"c{i}") for i in range(5)]

    start = time.monotonic()
    out = await node(ai_with_tool_calls(calls))
    elapsed = time.monotonic() - start

    assert len(out["messages"]) == 5
    assert elapsed < (5 * _DELAY) * 0.5
```

The test constructs a graph state with 5 tool calls, passes it directly to the tool node, and checks the results. The tool node is invoked as a regular async function — not through the full graph machinery.

This is the key insight: **individual graph nodes can be tested as regular async functions**. You do not need to build and invoke the full graph to test what a single node does.

### Building Tool Calls for Different Scenarios

```python
# A single tool call
calls = [tool_call("read_file", {"path": "src/main.py"}, "call_1")]

# Multiple tool calls (parallel)
calls = [
    tool_call("read_file", {"path": "a.py"}, "c1"),
    tool_call("read_file", {"path": "b.py"}, "c2"),
    tool_call("write_file", {"path": "out.py", "content": "..."}, "c3"),
]

# A blocked tool call
calls = [tool_call("write_file", {"path": "config.env", "content": "SECRET"}, "blocked")]

# An unknown tool call
calls = [tool_call("does_not_exist", {}, "unknown")]
```

The `tool_call()` helper makes constructing these states readable. Without the helper, you would write:

```python
{"name": "read_file", "args": {"path": "a.py"}, "id": "c1", "type": "tool_call"}
```

The helper names the positional arguments and hides the `"type": "tool_call"` boilerplate.

---

## 7. `structlog.testing.capture_logs`

Saathi uses `structlog` for structured logging. Instead of `logging.info("something happened, detail=%s", 42)`, it uses `log.info("something_happened", detail=42)`. The event is a string key; additional context is passed as keyword arguments.

Structured logging is easier to test than traditional logging because the log events are dictionaries, not formatted strings.

### `capture_logs` Context Manager

```python
from structlog.testing import capture_logs
```

`capture_logs()` is a context manager provided by structlog's testing support. Inside the `with` block, log events are captured into a list instead of being output to the console. After the block, you can assert on the captured events.

### Usage in `test_logging.py`

```python
def test_get_logger_emits_structured_event() -> None:
    with capture_logs() as logs:
        get_logger().warning("something_happened", detail=42)
    assert any(e["event"] == "something_happened" and e.get("detail") == 42 for e in logs)
```

`logs` is a list of dicts. Each dict represents one log event. The `"event"` key is the first argument to the log call. Additional keyword arguments are additional keys.

The `any(...)` assertion is deliberately permissive: it does not assert that exactly one log entry was emitted, or that it was the first. It just asserts that at least one entry with the right event and detail exists. This is robust to logging code that adds context or emits additional events.

### Asserting on Tool Behavior Through Logs

```python
async def test_tool_node_logs_block() -> None:
    node = make_hooked_tool_node(ALL_TOOLS, HookRunner(HookConfig(block_paths=["*.env"])))
    calls = [tool_call("write_file", {"path": "secret.env", "content": "x"}, "a")]
    with capture_logs() as logs:
        await node(ai_with_tool_calls(calls))
    assert any(e["event"] == "tool_blocked" and e.get("tool") == "write_file" for e in logs)
```

This test verifies that when a tool is blocked by the hook runner, the tool node logs a `"tool_blocked"` event with the tool name. It is testing the logging behavior — not just the outcome.

Why test logging? Because logging is part of the operational interface of the system. If the `tool_blocked` event stops being emitted, operators monitoring the log stream will not know that files are being blocked. The test is a contract: this event will be emitted when this behavior occurs.

```python
async def test_tool_node_logs_unknown_tool() -> None:
    node = make_hooked_tool_node(ALL_TOOLS, HookRunner(HookConfig()))
    with capture_logs() as logs:
        await node(ai_with_tool_calls([tool_call("does_not_exist", {}, "a")]))
    assert any(e["event"] == "tool_unknown" for e in logs)
```

Similarly, calling an unknown tool name logs `"tool_unknown"`. This is important: the agent might call a tool that does not exist (due to hallucination or a schema mismatch). The log event allows operators to detect and investigate this.

### What `capture_logs` Does Not Capture

`capture_logs()` captures events from structlog's processing pipeline. It does not capture Python's standard `logging` module, print statements, or rich console output. If your code mixes structlog and standard logging, you need separate mechanisms to capture each.

---

## 8. Testing the Graph

LangGraph's compiled graph is a runnable workflow — a complex object with nodes, edges, a checkpointer, and message routing. Testing the graph without Ollama requires building it with a fake LLM and asserting on its structure.

### Building the Graph in Tests

From `test_graph.py`:

```python
from pathlib import Path

from saathi.agent import build_graph, close_graph
from saathi.hooks.runner import HookRunner
from saathi.memory.store import MemoryStore
from saathi.tools import ALL_TOOLS


async def _build(tmp_path: Path):
    return await build_graph(
        ALL_TOOLS,
        MemoryStore(),
        "gemma4:12b",
        db_path=tmp_path / "checkpoints.db",
        hook_runner=HookRunner(),
    )
```

This builds a real compiled graph. The model ID `"gemma4:12b"` is passed, but the `ChatOllama` instance is not actually called during graph construction — it is only called during `graph.ainvoke()`. So building the graph is Ollama-free.

The `db_path=tmp_path / "checkpoints.db"` sends the SQLite checkpointer to the isolated temp directory.

### Testing Node Structure

```python
async def test_graph_compiles_with_expected_nodes(tmp_path: Path) -> None:
    graph = await _build(tmp_path)
    try:
        nodes = list(graph.get_graph().nodes)
        assert "agent" in nodes
        assert "tools" in nodes
    finally:
        await close_graph(graph)
```

`graph.get_graph()` returns a `Graph` object that describes the compiled graph's structure — nodes and edges — without running it. This is a pure inspection operation. It does not call the LLM or execute any tools.

The `try/finally` ensures `close_graph(graph)` is called even if the test fails. `close_graph` closes the SQLite connection. Without it, the temp directory cleanup might fail on Windows because SQLite holds a file lock.

### Testing the Checkpointer

```python
async def test_checkpoint_db_created(tmp_path: Path) -> None:
    db = tmp_path / "checkpoints.db"
    graph = await build_graph(
        ALL_TOOLS, MemoryStore(), "gemma4:12b", db_path=db, hook_runner=HookRunner()
    )
    try:
        assert db.exists()
    finally:
        await close_graph(graph)
```

This test verifies that the SQLite checkpointer creates its database file. It is testing infrastructure plumbing — a small thing that is nonetheless important to verify. If the `build_graph` function stops creating the checkpointer, this test fails immediately.

### Testing Thread Isolation

```python
async def test_fresh_thread_isolates_compacted_history(tmp_path: Path) -> None:
    """Compaction switches thread_id; the new thread must not inherit the old
    thread's accumulated messages."""
    graph = await _build(tmp_path)
    try:
        t1 = {"configurable": {"thread_id": "t1"}}
        await graph.aupdate_state(
            t1,
            {"messages": [
                HumanMessage(content="h1", id="1"),
                AIMessage(content="a1", id="2"),
                HumanMessage(content="h2", id="3"),
                AIMessage(content="a2", id="4"),
            ]},
            as_node="agent",
        )
        st1 = await graph.aget_state(t1)
        assert len(st1.values["messages"]) == 4

        t2 = {"configurable": {"thread_id": "t2"}}
        await graph.aupdate_state(
            t2,
            {"messages": [
                SystemMessage(content="summary", id="s"),
                HumanMessage(content="h2", id="3"),
                AIMessage(content="a2", id="4"),
            ]},
            as_node="agent",
        )
        st2 = await graph.aget_state(t2)
        assert [m.content for m in st2.values["messages"]] == ["summary", "h2", "a2"]
        assert len(st2.values["messages"]) == 3
    finally:
        await close_graph(graph)
```

This is a more complex graph test. It directly manipulates graph state using `aupdate_state` — a way to inject messages into the graph's checkpointer without running the graph.

The test verifies that threads are isolated: state written to thread `t1` does not appear in thread `t2`. This is a crucial property of the compaction system. Compaction works by creating a new thread with a condensed history. If the new thread inherited the old thread's history, the compaction would not shrink the context.

### The `close_graph` Pattern

Every `test_graph.py` test follows the pattern:

```python
graph = await _build(tmp_path)
try:
    # ... test assertions ...
finally:
    await close_graph(graph)
```

The `try/finally` is essential. SQLite's async connection must be explicitly closed. If a test fails and raises an exception, the `finally` block ensures the connection is still closed. Without this, the test suite would accumulate open database connections and might fail with "database is locked" errors on Windows.

This is a general pattern for resource management in async tests: acquire → test → release in `finally`.

---

## 9. The Regression Guard Pattern

A regression guard is a test that exists not to verify a feature works, but to prevent a specific known bug from being reintroduced. The test documents the bug and ensures it cannot silently reappear.

### The Bug

At some point in saathi's development, a `builder.add_edge("agent", END)` call was added to the graph builder. This seems reasonable — it says "after the agent runs, end the workflow." But LangGraph already has a conditional router (`tools_condition`) that routes `agent → tools` when there are tool calls and `agent → END` when there are none.

Adding an unconditional `agent → END` edge breaks the ReAct loop: the graph routes to END after every agent step, even when the agent wanted to call tools. The agent calls a tool, but the result never comes back.

The bug is hard to catch by reading the graph building code. It requires understanding LangGraph's conditional edge semantics. And it can be reintroduced easily — it looks like a reasonable line of code.

### The Guard Test

```python
async def test_agent_only_ends_conditionally(tmp_path: Path) -> None:
    """Regression: the agent node must not have an unconditional edge to END.

    tools_condition already routes agent -> tools or agent -> END. An extra
    static agent -> END edge would terminate the ReAct loop on every step.
    """
    graph = await _build(tmp_path)
    try:
        edges = graph.get_graph().edges
        agent_out = [e for e in edges if e.source == "agent"]
        # Both outgoing edges (to tools and to END) must be conditional.
        assert agent_out, "agent has no outgoing edges"
        assert all(e.conditional for e in agent_out), (
            "agent has an unconditional edge; the ReAct loop would break"
        )
    finally:
        await close_graph(graph)
```

This test:

1. Compiles the graph
2. Extracts all edges where the source node is `"agent"`
3. Asserts that every outgoing edge is conditional

The LangGraph `Graph.edges` property returns edge objects with a `.conditional` attribute. A conditional edge has a routing function; an unconditional edge goes directly to its destination.

The error message in the assertion — "agent has an unconditional edge; the ReAct loop would break" — is the documentation. It tells the future developer exactly why this test exists and what will break if the assertion fails.

### Writing Good Regression Guards

A good regression guard:

1. **Names the bug in the docstring.** "Regression: ..." tells readers this test is not testing normal functionality — it is a bug prevention measure.
2. **Explains why the bug breaks the system.** The test should document what would actually fail in production, not just that the test fails.
3. **Checks the specific invariant, not the symptom.** This test checks for unconditional edges, not for "agent calls tools correctly." The invariant is structural; the symptom would be functional.
4. **Is easy to understand in isolation.** A developer who has never read the original bug report should be able to understand the test and its assertion.

### Other Candidates for Regression Guards

Any time you fix a bug that was not caught by an existing test, consider writing a regression guard:

- A validation rule that was missing and allowed invalid data
- A concurrency issue that occurred under a specific sequence of operations
- An off-by-one in a splitting or slicing operation
- A missing `finally` that left resources open

The investment is small (a few lines of test code) and the return is permanent: the bug cannot be silently reintroduced.

---

## 10. `monkeypatch` for Environment Variables

Saathi's configuration is driven by `pydantic-settings`, which reads from environment variables with a `SAATHI_` prefix. Tests need to verify that configuration overrides work correctly.

The `monkeypatch` pytest fixture provides safe, test-scoped mutation of environment variables, object attributes, and module-level variables.

### Testing Environment Variable Overrides

From `test_config.py`:

```python
def test_env_prefix_override(monkeypatch) -> None:
    monkeypatch.setenv("SAATHI_MAX_PARALLEL_TOOLS", "3")
    monkeypatch.setenv("SAATHI_OLLAMA_MODEL", "llama3:8b")
    s = Settings(_env_file=None)
    assert s.max_parallel_tools == 3
    assert s.ollama_model == "llama3:8b"
```

`monkeypatch.setenv("KEY", "VALUE")` sets an environment variable for the duration of the test. After the test completes (whether it passes or fails), the environment variable is automatically unset. There is no cleanup code needed.

The `Settings(_env_file=None)` creates a fresh `Settings` instance without reading from `.env` files — only from the current environment. This ensures the test is not affected by any `.env` file on the developer's machine.

### Testing Default Values

```python
def test_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.ollama_model
    assert s.ollama_base_url.startswith("http")
    assert s.max_parallel_tools == 8
    assert 0.0 <= s.temperature <= 2.0
```

This test does not use `monkeypatch` because it tests the default values — what you get when no environment variables are set. Note the deliberate choice to test a range for `temperature` (`0.0 <= s.temperature <= 2.0`) rather than an exact value. The exact default temperature might change; that it is a valid temperature value should not.

### Testing Derived Properties

```python
def test_history_budget_is_75_percent() -> None:
    s = Settings(_env_file=None)
    assert s.history_token_budget == int(s.context_window * 0.75)
```

`history_token_budget` is a `@property` that computes `int(context_window * 0.75)`. This test verifies the formula, not a specific value. If `context_window` changes, the test still passes as long as the ratio is correct.

### `monkeypatch.setattr` for Object Attributes

`monkeypatch` is not limited to environment variables. You can patch any attribute of any object:

```python
async def test_semaphore_caps_concurrency(monkeypatch) -> None:
    from saathi.config import settings
    monkeypatch.setattr(settings, "max_parallel_tools", 2)
    # ... test that at most 2 tools run concurrently ...
```

This directly patches the `settings` singleton's `max_parallel_tools` attribute to `2` for the duration of the test. After the test, it reverts to the original value.

This is safer than `monkeypatch.setenv` for this test because patching the attribute directly bypasses `pydantic-settings`'s validation and parsing. It is the right choice when you want to test code that reads from `settings.max_parallel_tools` without going through the full settings construction.

### Why Not Just Set Environment Variables Directly?

```python
import os
os.environ["SAATHI_MAX_PARALLEL_TOOLS"] = "3"  # DON'T do this
```

Setting `os.environ` directly in a test is dangerous because:

1. If the test fails before cleanup, the environment variable remains set, affecting subsequent tests.
2. If multiple tests run in parallel (with `pytest-xdist`), they will interfere with each other.
3. It requires explicit cleanup code that is easy to forget.

`monkeypatch.setenv` is automatically scoped to the test's lifetime. It is always the right choice over `os.environ` manipulation in tests.

---

## 11. Testing Parallel Execution

Saathi's tool node runs tool calls concurrently using `asyncio`. Verifying that concurrency actually works requires a timing-based test.

### The Timing Test

From `test_parallel.py`:

```python
_DELAY = 0.3


@tool
async def slow_echo(label: str) -> str:
    """Sleep then echo the label back."""
    await asyncio.sleep(_DELAY)
    return f"done:{label}"


async def test_calls_run_in_parallel() -> None:
    node = make_hooked_tool_node([slow_echo], _no_hooks())
    calls = [tool_call("slow_echo", {"label": str(i)}, f"c{i}") for i in range(5)]

    start = time.monotonic()
    out = await node(ai_with_tool_calls(calls))
    elapsed = time.monotonic() - start

    assert len(out["messages"]) == 5
    # 5 serial calls would take ~5*_DELAY; parallel should be well under half that.
    assert elapsed < (5 * _DELAY) * 0.5
    assert sorted(m.content for m in out["messages"]) == [f"done:{i}" for i in range(5)]
```

Five calls, each sleeping 0.3 seconds:

- **Serial execution:** 5 × 0.3s = 1.5s
- **Parallel execution:** max(0.3s) ≈ 0.3s

The assertion `elapsed < (5 * _DELAY) * 0.5` means "less than 0.75 seconds." This is generous: the actual parallel execution takes about 0.3 seconds, plus some overhead for event loop scheduling, tool dispatch, and message construction. The 0.75-second threshold gives substantial headroom.

### Why a Generous Upper Bound?

Timing-based tests are inherently susceptible to system load. If the CI runner is under heavy load, the test might take longer. Tight timing assertions (`elapsed < 0.35`) would produce flaky tests.

The rule is: **use the slowest plausible serial time, then take half.** Serial time for this test is `5 * 0.3s = 1.5s`. Half is `0.75s`. Parallel execution should easily finish in under `0.75s`.

If the test fails, it means the execution took more than 0.75 seconds — more than half the serial time. That is a clear indication of sequential execution, not random system load.

### The Output Completeness Test

```python
async def test_every_tool_call_is_answered() -> None:
    node = make_hooked_tool_node([slow_echo], _no_hooks())
    calls = [tool_call("slow_echo", {"label": str(i)}, f"id{i}") for i in range(4)]
    out = await node(ai_with_tool_calls(calls))
    assert {m.tool_call_id for m in out["messages"]} == {f"id{i}" for i in range(4)}
```

Every tool call has a unique `tool_call_id`. Every response message should have a corresponding `tool_call_id`. This test verifies that no tool calls are dropped — all four calls produce responses.

This is a correctness test, not a performance test. It verifies that parallel execution does not silently drop results.

### The Semaphore Test

```python
async def test_semaphore_caps_concurrency(monkeypatch) -> None:
    from saathi.config import settings
    monkeypatch.setattr(settings, "max_parallel_tools", 2)

    state = {"active": 0, "peak": 0}
    lock = asyncio.Lock()

    @tool
    async def tracked(i: int) -> str:
        """Track how many copies run at once."""
        async with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.05)
        async with lock:
            state["active"] -= 1
        return str(i)

    node = make_hooked_tool_node([tracked], _no_hooks())
    calls = [tool_call("tracked", {"i": i}, f"c{i}") for i in range(6)]
    await node(ai_with_tool_calls(calls))

    assert state["peak"] <= 2
```

This test verifies that the semaphore (`max_parallel_tools=2`) actually limits concurrency. The `tracked` tool uses an asyncio lock to safely count how many instances are active simultaneously. After all calls complete, `state["peak"]` holds the maximum concurrent count.

With `max_parallel_tools=2`, no more than 2 instances should run simultaneously. `assert state["peak"] <= 2` verifies this.

The `asyncio.Lock()` is required because the counter updates are not atomic. Without the lock, two coroutines could read `state["active"]` before either increments it, causing a race condition in the tracking code itself — which would make the test unreliable. The irony of needing a lock to test that a lock works is real.

---

## 12. Testing Hooks

Hooks are shell scripts that run before and after tool calls. Testing them requires careful handling of real subprocess execution.

### Testing the Block List

Block-list testing does not require shell execution — it is pure Python string matching:

```python
def test_check_block_matches_env() -> None:
    runner = HookRunner(HookConfig(block_paths=["*.env"]))
    assert runner.check_block("write_file", {"path": "config.env"})
    assert runner.check_block("patch_file", {"path": "config.env"})


def test_check_block_only_write_tools() -> None:
    runner = HookRunner(HookConfig(block_paths=["*.env"]))
    assert runner.check_block("read_file", {"path": "config.env"}) is None
    assert runner.check_block("write_file", {"path": "notes.txt"}) is None
```

`check_block` returns a non-None value (the reason for blocking) when a write tool is called on a path that matches the block pattern, and `None` otherwise. These tests verify the matching logic without running any tools.

### Testing Pre-Tool Hook Behavior

```python
async def test_pre_tool_hook_blocks_on_nonzero_exit(sample_file: Path) -> None:
    runner = HookRunner(HookConfig(pre_tool=["exit 1"]))
    node = make_hooked_tool_node(ALL_TOOLS, runner)
    calls = [tool_call("read_file", {"path": str(sample_file)}, "x")]
    out = await node(ai_with_tool_calls(calls))
    assert out["messages"][0].content.startswith("BLOCKED")
    assert "pre_tool hook rejected" in out["messages"][0].content
```

The pre-tool hook `["exit 1"]` is a shell command that exits with code 1. When the hook runner runs this command before a tool call and gets exit code 1, it blocks the tool call. The test verifies that the tool node returns a "BLOCKED" message.

The `sample_file` fixture provides a real file path that the `read_file` tool would normally read successfully. The hook rejection happens before the tool runs, so the file is never actually read.

```python
async def test_pre_tool_hook_allows_on_zero_exit(sample_file: Path) -> None:
    runner = HookRunner(HookConfig(pre_tool=["exit 0"]))
    node = make_hooked_tool_node(ALL_TOOLS, runner)
    calls = [tool_call("read_file", {"path": str(sample_file)}, "x")]
    out = await node(ai_with_tool_calls(calls))
    assert out["messages"][0].content == "SAMPLE_CONTENT_123"
```

The companion test: `exit 0` means success, the hook allows the tool, and the tool returns the file content.

### Testing Post-Tool Hook Execution

```python
async def test_post_tool_hook_executes(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    runner = HookRunner(HookConfig(post_tool=[f'echo hooked > "{marker}"']))
    results = await runner.run("post_tool", "read_file", {"path": "x"})
    assert results and results[0].ok
    assert marker.exists()
```

This test writes a real file as a side effect of running the hook. The hook script uses `echo hooked > "path"` to create a file. The test then asserts that the file exists.

Using `tmp_path` here is important — the marker file goes into the isolated temp directory, not the project root.

### Testing Cross-Platform Hook Scripts

```python
async def test_hook_env_vars_available(tmp_path: Path) -> None:
    marker = tmp_path / "env.txt"
    var = "%SAATHI_TOOL_NAME%" if os.name == "nt" else "$SAATHI_TOOL_NAME"
    runner = HookRunner(HookConfig(post_tool=[f'echo {var} > "{marker}"']))
    await runner.run("post_tool", "write_file", {"path": "p"})
    assert "write_file" in marker.read_text(encoding="utf-8")
```

The hook runner injects the tool name as an environment variable (`SAATHI_TOOL_NAME`). The hook script accesses it with `$SAATHI_TOOL_NAME` on Unix or `%SAATHI_TOOL_NAME%` on Windows.

The `os.name == "nt"` conditional is necessary because the hook scripts run in the system's shell — `cmd.exe` on Windows, `sh` on Unix. Environment variable syntax differs between them. This is one of the few places in the test suite where platform-specific code is appropriate.

### Testing Config Loading

```python
def test_load_example_config() -> None:
    cfg = load_hook_config(Path("hooks.example.json"))
    assert "*.env" in cfg.block_paths
    assert cfg.post_turn
    assert not cfg.is_empty


def test_load_missing_config_is_empty(tmp_path: Path) -> None:
    cfg = load_hook_config(tmp_path / "absent.json")
    assert cfg.is_empty


def test_load_malformed_config_is_empty(tmp_path: Path) -> None:
    bad = tmp_path / "hooks.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    assert load_hook_config(bad).is_empty
```

These tests verify that `load_hook_config` handles all three cases gracefully: valid config, missing config, and malformed config. The "graceful" behavior for missing and malformed config is to return an empty (no-op) `HookConfig` rather than raising.

---

## 13. `pytest.mark.live` — Opt-in E2E Tests

End-to-end tests that require a real running Ollama instance are valuable but cannot run in CI. Saathi solves this with a pytest mark that excludes them by default.

### The `live` Mark

In `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "-ra -q -m 'not live'"
markers = [
    "live: end-to-end test that requires a running Ollama (run with `pytest -m live`)",
]
```

`addopts = "-m 'not live'"` means "by default, run all tests except those marked `live`." The default `pytest` run never touches live tests.

The `markers` list registers the custom mark. Without this, pytest would warn about unknown marks.

### Applying the Mark

From `test_e2e_live.py`:

```python
import pytest

pytestmark = pytest.mark.live
```

The module-level `pytestmark` applies the `live` mark to every test in the file. All tests in `test_e2e_live.py` are automatically live tests, without needing `@pytest.mark.live` on each function.

This is the right pattern when an entire test file is live — not just one test. It is cleaner and harder to accidentally miss.

### Running Live Tests

```bash
# Run only live tests
pytest -m live

# Run all tests including live
pytest -m ''

# Run live tests with verbose output
pytest -m live -v
```

### The Live Test Structure

```python
async def test_live_single_turn(tmp_path: Path) -> None:
    if not _ollama_ready():
        pytest.skip(f"Ollama not reachable at {settings.ollama_base_url} or model missing")

    graph = await build_graph(
        ALL_TOOLS,
        MemoryStore(),
        settings.ollama_model,
        db_path=tmp_path / "cp.db",
        hook_runner=HookRunner(),
    )
    try:
        config = {"configurable": {"thread_id": "e2e"}}
        state = {
            "messages": [HumanMessage(content="Reply with exactly the word: PONG")],
            "context_paths": [],
            "mode": "default",
            "session_id": "e2e",
        }
        result = await graph.ainvoke(state, config)
        answers = [
            m
            for m in result["messages"]
            if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip()
        ]
        assert answers, "expected at least one assistant response"
        assert answers[-1].content.strip()
    finally:
        await close_graph(graph)
```

The test uses a deliberate prompt: "Reply with exactly the word: PONG." This prompts a short, specific response. The assertion is generous: "the agent produced at least one non-empty response." It does not assert the exact content — models sometimes respond "PONG!", "Pong", or "The word is PONG." All are acceptable.

The purpose of this test is to verify that:

1. The graph builds and compiles correctly
2. LangGraph routes the turn through the graph
3. The agent node invokes the LLM successfully
4. The LLM produces a response
5. The response flows back through the graph

It is a smoke test, not a behavior test.

---

## 14. The Ollama Readiness Check

The e2e test includes a readiness check before running. This check verifies that Ollama is running and the required model is available, and skips the test rather than failing if either condition is not met.

```python
def _ollama_ready() -> bool:
    """True if Ollama is reachable and the configured model is available."""
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return False
    want = settings.ollama_model
    return any(m == want or m.split(":")[0] == want.split(":")[0] for m in models)
```

### What It Checks

1. **Network reachability:** Can we reach `{ollama_base_url}/api/tags`? A `timeout=3` prevents the check from hanging for long.
2. **Model availability:** Is the configured model (`settings.ollama_model`) in the list of pulled models?

The model check uses flexible matching:

```python
any(m == want or m.split(":")[0] == want.split(":")[0] for m in models)
```

Ollama model names include a tag (`gemma4:12b`, `llama3:8b`). The check tries exact match first, then tag-agnostic match (comparing just the model family name before `:`). This means that if `settings.ollama_model = "gemma4"` but Ollama has `"gemma4:12b"` pulled, the check succeeds.

### `pytest.skip` vs `pytest.xfail`

The test uses `pytest.skip(reason)` rather than `pytest.xfail(reason)`. The difference:

- `pytest.skip`: The test is not run. It appears as "s" in the output. This is the right choice for infrastructure reasons — the test is not failing; the environment does not support it.
- `pytest.xfail`: The test is expected to fail. It is still run. This is for bugs you know about but have not fixed yet.

When Ollama is not available, skipping is correct. The test did not fail; it could not run.

### Using Skipif at Import Time

An alternative pattern is `@pytest.mark.skipif`:

```python
@pytest.mark.skipif(not _ollama_ready(), reason="Ollama not available")
async def test_live_single_turn(tmp_path: Path) -> None:
    ...
```

The difference: `skipif` at import time calls `_ollama_ready()` when the test module is imported, not when the test runs. If collection takes a long time, or if Ollama becomes available after collection, this produces incorrect results.

The internal readiness check (inside the test body) is more accurate: it checks exactly when the test is about to run. The trade-off is a slightly more verbose test body.

---

## 15. Testing MCP with a Real Server

The MCP (Model Context Protocol) client connects to external servers over stdio or HTTP. Testing this without external dependencies requires spawning a local test server process.

### The Echo Server Pattern

Saathi includes `examples/mcp_echo_server.py` — a minimal MCP server that implements a single `echo` tool. It is a real MCP server, written in pure Python, that can be spawned as a subprocess.

From `test_mcp.py`:

```python
_ECHO_SERVER = Path("examples/mcp_echo_server.py")


@pytest.mark.skipif(not _ECHO_SERVER.is_file(), reason="echo server example missing")
async def test_echo_server_tool_roundtrip() -> None:
    conns = {"echo": {"transport": "stdio", "command": sys.executable, "args": [str(_ECHO_SERVER)]}}
    tools = await load_mcp_tools(conns)
    names = {t.name for t in tools}
    assert "echo" in names
    echo = next(t for t in tools if t.name == "echo")
    result = await echo.ainvoke({"text": "hi"})
    assert "echo: hi" in str(result)
```

`sys.executable` is the path to the Python interpreter running the test. Using it ensures the echo server is launched with the same Python and the same virtual environment as the test, avoiding dependency mismatch.

The connection config uses `transport: "stdio"` — the MCP server communicates over stdin/stdout. `load_mcp_tools` spawns the process, connects to it via the MCP protocol, and retrieves the tool list.

This is a real round-trip: the test calls an MCP tool, the call is serialized to MCP protocol, sent to the subprocess over stdin, the subprocess executes the tool, and the result is returned over stdout. No mocking.

### Offline MCP Tests

The rest of `test_mcp.py` tests the config loading and normalization logic without spawning any servers:

```python
def test_mcp_servers_wrapper_and_stdio_inference(tmp_path: Path) -> None:
    p = tmp_path / "mcp.json"
    p.write_text(
        json.dumps({"mcpServers": {"fs": {"command": "npx", "args": ["-y", "x"]}}}),
        encoding="utf-8",
    )
    conns = load_mcp_config(p)
    assert conns == {"fs": {"transport": "stdio", "command": "npx", "args": ["-y", "x"]}}


def test_flat_mapping_and_url_inference(tmp_path: Path) -> None:
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({"weather": {"url": "http://localhost:8000/mcp/"}}), encoding="utf-8")
    conns = load_mcp_config(p)
    assert conns["weather"]["transport"] == "streamable_http"
    assert conns["weather"]["url"] == "http://localhost:8000/mcp/"
```

These tests write various MCP config JSON shapes to temp files and verify that `load_mcp_config` normalizes them correctly — inferring the transport type, unwrapping the `mcpServers` wrapper, and handling URL vs. command-based servers.

### Graceful Degradation Test

```python
async def test_unreachable_server_degrades_gracefully() -> None:
    conns = {
        "nope": {"transport": "stdio", "command": "definitely-not-a-real-binary-xyz", "args": []}
    }
    assert await load_mcp_tools(conns) == []
```

An MCP server that cannot be spawned (the command does not exist) should produce zero tools, not an exception. This tests that `load_mcp_tools` catches subprocess errors and returns gracefully.

This is the same graceful degradation principle as the review system's `review_one`: external failures should produce empty results, not crashes.

---

## 16. CI Configuration

A well-configured CI pipeline is the enforcement mechanism for the test suite. Tests that are not run in CI are tests that can fail without anyone noticing.

### The GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    strategy:
      matrix:
        python-version: ["3.12", "3.13"]

    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install hatch
          hatch env create

      - name: Lint (ruff)
        run: hatch run ruff check src tests

      - name: Type-check (mypy)
        run: hatch run mypy src

      - name: Test (pytest)
        run: hatch run pytest
        # Note: addopts in pyproject.toml excludes live tests automatically.
        # No Ollama in CI — live tests are excluded.
```

### The Matrix

Testing on Python 3.12 and 3.13 catches:

- `asyncio` behavior differences between versions
- Deprecated API usage that may be removed in a future version
- New syntax features accidentally used that are not backward-compatible

The matrix doubles the CI time but catches a class of bugs that single-version CI misses.

### Why No Ollama in CI

The GitHub Actions `ubuntu-latest` runner has a CPU but no GPU. Running a 12B parameter model on a CPU would take several minutes per inference call. A 4-reviewer code review would take 30–60 minutes.

More practically: Ollama requires downloading model weights (gigabytes), which would consume significant CI bandwidth and time. The live tests are specifically designed to be opt-in for this reason.

### Linting and Type Checking First

The workflow runs `ruff check` before `pytest`. This is intentional: if the code does not pass linting, there is no point running the tests. Linting fails fast.

Mypy type checking is also run before pytest. Type errors are not necessarily test failures, but they often reveal bugs. Running mypy in CI ensures type annotations stay accurate.

### Fast CI Feedback

The offline test suite should complete in under 10 seconds. This keeps the CI loop tight: push → CI triggered → results in under 5 minutes. Developers receive feedback while the change is still fresh in their minds.

---

## 17. Coverage

Code coverage measures what percentage of source lines are executed during the test suite.

### Running Coverage

```bash
pytest --cov=saathi --cov-report=term-missing
```

This runs the test suite with coverage tracking. `--cov=saathi` tracks coverage for the `saathi` package (source files in `src/saathi/`). `--cov-report=term-missing` shows which lines were not covered.

### What Good Coverage Looks Like for LLM Apps

For traditional applications, 90%+ coverage is achievable and meaningful. For LLM applications, the target is lower.

**80%+ is a realistic goal.** The remaining 20% typically consists of:

1. **LLM-dependent code paths.** Error handling for LLM failures, fallback behavior for malformed LLM responses, edge cases in the agent loop that only occur with specific model behavior. These are difficult to trigger with fake LLMs without over-engineering the fakes.

2. **CLI interaction code.** The CLI loop handles user input in ways that are difficult to simulate in tests without running the full CLI. Interactive features like the history viewer, paste mode, and rollback prompts require a TTY.

3. **Platform-specific branches.** Code that behaves differently on Windows vs. Unix (like the hook env var syntax test) may have one branch not covered on the CI platform.

4. **Error recovery code.** Truly exceptional conditions — out of memory, permission denied on the checkpointer database, corrupted SQLite file — are difficult to trigger in tests.

### What Coverage Means (and Does Not Mean)

Coverage tells you which lines were executed. It does not tell you whether the assertions tested those lines meaningfully. A test can achieve 100% coverage of a function while testing nothing meaningful about its behavior.

Use coverage to find obvious gaps: untouched files, untouched functions, large untouched branches. Do not use it as the primary quality metric.

The most useful coverage metric is **branch coverage** (did both branches of each `if` statement execute?), which ruff and mypy's exhaustiveness checking also help with.

### Coverage for Specific Modules

```bash
# Show coverage for a specific file
pytest --cov=saathi.review --cov-report=term-missing tests/test_review.py

# Generate HTML report for detailed line-by-line view
pytest --cov=saathi --cov-report=html
# Then open htmlcov/index.html
```

The HTML report shows exact line highlighting — green for covered, red for not covered. It is useful for identifying specific uncovered code paths.

---

## 18. Property-Based Testing with Hypothesis

Property-based testing generates random inputs and looks for inputs that violate stated properties. Instead of writing specific test cases, you write invariants.

### Introduction to Hypothesis

```bash
pip install hypothesis
```

```python
from hypothesis import given, strategies as st


@given(st.integers())
def test_confidence_clamping_never_out_of_range(n: int) -> None:
    f = Finding(confidence=n)
    assert 0 <= f.confidence <= 100
```

Hypothesis generates hundreds of random integers — including edge cases like `0`, `1`, `-1`, `2**31 - 1`, `2**63`, `-2**63` — and verifies that `Finding(confidence=n).confidence` is always in `[0, 100]`. If any integer violates the invariant, Hypothesis reports the minimal failing case.

### Where Hypothesis Applies in Saathi

**Token estimation:** `estimate_tokens` computes `total_chars // 4`. For any list of messages, the estimate should be non-negative:

```python
from hypothesis import given, strategies as st
from langchain_core.messages import HumanMessage
from saathi.compaction import estimate_tokens


@given(st.lists(st.text()))
def test_estimate_tokens_is_nonnegative(texts: list[str]) -> None:
    messages = [HumanMessage(content=t) for t in texts]
    assert estimate_tokens(messages) >= 0
```

**JSON extraction:** `_extract_json` should never raise, regardless of input:

```python
from hypothesis import given, strategies as st
from saathi.review import _extract_json


@given(st.text())
def test_extract_json_never_raises(text: str) -> None:
    result = _extract_json(text)
    assert result is None or isinstance(result, (dict, list))
```

This property is simple but powerful: throw any string at `_extract_json` and it should always return either `None`, a dict, or a list — never raise an exception.

**Finding validation:** For any string, severity normalization should produce a valid severity:

```python
@given(st.text())
def test_severity_always_valid(s: str) -> None:
    f = Finding(severity=s)
    assert f.severity in ("high", "medium", "low")
```

### Where Hypothesis Does NOT Apply

Hypothesis is not useful for:

- Tests that require specific API responses (LLM calls, external services)
- Tests where the "property" is essentially "the function returns the right thing" (that is what unit tests are for)
- Tests where the input space is naturally constrained and the edge cases are known

Property-based testing works best for pure functions with well-defined invariants: parsers, validators, transformations, and utility functions.

---

## 19. Test Organization

A well-organized test suite is easier to maintain, navigate, and understand than a monolithic or scattered one.

### Saathi's Structure

```folder
tests/
├── __init__.py
├── conftest.py          # Shared fixtures
├── helpers.py           # Plain helper functions (not fixtures)
├── test_compaction.py   # Tests for src/saathi/compaction.py
├── test_config.py       # Tests for src/saathi/config.py
├── test_custom_commands.py  # Tests for src/saathi/custom_commands.py
├── test_diagnostics.py  # Tests for src/saathi/diagnostics.py
├── test_e2e_live.py     # Live e2e tests (marked `live`)
├── test_git.py          # Tests for src/saathi/tools/git.py
├── test_graph.py        # Tests for src/saathi/agent/graph.py
├── test_hooks.py        # Tests for src/saathi/hooks/runner.py
├── test_imports.py      # Smoke test: all modules import without error
├── test_logging.py      # Tests for src/saathi/logging_config.py
├── test_mcp.py          # Tests for src/saathi/mcp_client.py
├── test_memory.py       # Tests for src/saathi/memory/store.py
├── test_parallel.py     # Tests for parallel tool execution
├── test_print_mode.py   # Tests for print/non-interactive mode
├── test_project_context.py  # Tests for src/saathi/project_context.py
├── test_prompts.py      # Tests for src/saathi/agent/prompts.py
├── test_retry.py        # Tests for src/saathi/retry.py
├── test_review.py       # Tests for src/saathi/review.py
├── test_snapshots.py    # Tests for snapshot/rollback behavior
├── test_tools.py        # Tests for src/saathi/tools/
└── test_usage.py        # Tests for usage tracking
```

### One Test File Per Module

Each source module has a corresponding test file. The mapping is intentional and enforced by convention, not tooling. It makes it easy to find the tests for a given module and to know where to add new tests when adding functionality.

The exception is `test_parallel.py`, which tests behavior that spans `tool_node.py` and `config.py` together. Some behaviors are architectural rather than module-specific, and those get their own test files.

### `conftest.py` — Shared Fixtures

`conftest.py` contains fixtures that are shared across multiple test files. pytest automatically loads it before running tests.

```python
# tests/conftest.py

from pathlib import Path

import pytest

from saathi.memory.store import MemoryStore


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """A small readable file with known content, isolated in tmp_path."""
    p = tmp_path / "sample.txt"
    p.write_text("SAMPLE_CONTENT_123", encoding="utf-8")
    return p


@pytest.fixture
def isolated_memory(tmp_path: Path) -> MemoryStore:
    """A MemoryStore whose global/project files live under tmp_path (no real writes)."""
    store = MemoryStore()
    store._global_path = tmp_path / "global" / "memory.json"
    store._project_path = tmp_path / "project" / "memory.json"
    return store
```

Saathi's `conftest.py` is deliberately small: only two fixtures. Fixtures that are used by only one test file belong in that file, not in `conftest.py`. The more things in `conftest.py`, the harder it is to understand what a test depends on.

### `helpers.py` — Plain Helper Functions

`helpers.py` contains plain functions (not fixtures) used across multiple test files:

```python
# tests/helpers.py

from langchain_core.messages import AIMessage


def ai_with_tool_calls(calls: list[dict]) -> dict:
    """Build a graph state whose last message carries the given tool calls."""
    return {
        "messages": [AIMessage(content="", tool_calls=calls)],
        "context_paths": [],
        "mode": "default",
        "session_id": "test",
    }


def tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}
```

The distinction between `conftest.py` (fixtures) and `helpers.py` (plain functions) is important. pytest fixtures have special semantics: they are injected by name, they can have scopes, they support setup and teardown. Plain helper functions are just functions — imported and called normally.

Using a fixture for a plain helper would add unnecessary indirection. A function that builds a state dict is just a function; it does not need the fixture machinery.

### The Import Smoke Test

```python
# tests/test_imports.py

def test_all_modules_import() -> None:
    import saathi
    import saathi.agent
    import saathi.cli
    import saathi.compaction
    import saathi.config
    import saathi.diagnostics
    import saathi.hooks.runner
    import saathi.logging_config
    import saathi.mcp_client
    import saathi.memory.store
    import saathi.project_context
    import saathi.retry
    import saathi.review
    import saathi.session.manager
    import saathi.tools
    import saathi.ui.display
```

This test simply imports every module. A module that fails to import causes all tests in that module to fail with `ImportError` — but the error messages are sometimes unclear. This test isolates import failures explicitly.

It catches: circular imports, missing dependencies, syntax errors in module-level code, and initialization errors (like database connections opened at module level).

### Test Naming Conventions

- `test_<module>_<behavior>` for tests about a specific module's behavior
- `test_<behavior>` when the test is about a cross-cutting behavior
- Avoid abbreviations: `test_confidence_is_clamped` is clearer than `test_conf_clamp`
- Verb form: tests describe what the system does, not what you are checking

### Grouping Tests with Comments

For test files with many tests, group them with section comments:

```python
# ── Finding validation ────────────────────────────────────────────────────────
def test_confidence_is_clamped() -> None: ...
def test_severity_is_normalized() -> None: ...
def test_line_is_coerced() -> None: ...


# ── JSON extraction ───────────────────────────────────────────────────────────
def test_extract_json_plain_and_fenced() -> None: ...


# ── parse_findings ────────────────────────────────────────────────────────────
def test_parse_object_and_array_forms() -> None: ...
def test_parse_skips_non_dict_and_garbage() -> None: ...


# ── run_review aggregation / filtering / ranking ──────────────────────────────
async def test_run_review_filters_below_confidence() -> None: ...
async def test_run_review_sorts_by_severity_then_confidence() -> None: ...
```

The section comments are a lightweight alternative to test classes. They provide organization without the overhead of `self` parameters and class setup/teardown.

---

## 20. The "Test Doubles" Vocabulary

The term "fake LLM" is informal. The formal vocabulary of test doubles provides useful distinctions:

### Stub

A stub returns predetermined responses without recording how it was called. It answers questions but does not care what questions it was asked.

```python
class StubLLM:
    async def ainvoke(self, messages, config=None):
        return AIMessage(content='{"findings": []}')
```

Use stubs when you care about the code's behavior after receiving the response, not about what was sent to the LLM.

In saathi's test suite: `FakeLLM` with a fixed response and no call recording is essentially a stub (though it also counts calls, making it slightly more than a stub).

### Mock

A mock records how it was called and can assert on those calls. It is used when the test needs to verify that the code under test called the mock with the right arguments.

```python
from unittest.mock import AsyncMock

mock_llm = AsyncMock()
mock_llm.ainvoke.return_value = AIMessage(content='{"findings": []}')

# ... run code under test ...

mock_llm.ainvoke.assert_called_once()
args = mock_llm.ainvoke.call_args[0][0]
assert isinstance(args[0], SystemMessage)
```

Use mocks when you need to assert on how the LLM was invoked (message content, number of calls, specific arguments).

### Fake

A fake is a working implementation of a component that is simpler than the real thing. It implements the same interface but uses shortcuts that make it unsuitable for production.

```python
class FakeMemoryStore:
    """In-memory implementation of MemoryStore, no file I/O."""

    def __init__(self) -> None:
        self._data: dict = {}

    def load(self, key: str) -> dict:
        return self._data.get(key, {})

    def save(self, key: str, value: dict) -> None:
        self._data[key] = value
```

A fake `MemoryStore` uses an in-memory dict instead of reading and writing JSON files. It is correct for testing (reads return what was written), but it does not persist data.

The compaction `FakeLLM` is a fake: it actually implements the `ainvoke` interface and returns a proper `AIMessage`. It is just simplified (fixed response rather than model-generated).

### Spy

A spy wraps the real implementation and records calls, passing them through to the real implementation.

```python
class SpyLLM:
    def __init__(self, real_llm):
        self._real = real_llm
        self.calls: list = []

    async def ainvoke(self, messages, config=None):
        self.calls.append(messages)
        return await self._real.ainvoke(messages, config)
```

Spies are useful when you need to test that code calls a component correctly, while still needing the component's real behavior. In LLM testing, spies are rarely useful — you almost always want a fake response, not a real one.

### Which to Use in LLM Testing

| Scenario | Double Type |
| --- | --- |
| Test code that processes LLM responses | Stub (fixed response) |
| Verify the LLM was called the right number of times | Stub with call counter |
| Verify the LLM was called with specific messages | Mock or Spy |
| Test a component that uses LLM internally | Fake (full interface, simplified behavior) |
| Test that the LLM produces correct output | Real LLM (live test) |

The most important choice: **do not use real LLMs for unit tests, ever.** The value of a fast, offline test suite depends on it staying fast and offline. Every real LLM call in a unit test is a violation of that contract.

---

## Summary

Testing LLM applications well requires:

1. **Accepting the non-determinism problem** — you cannot test what the LLM says, but you can test everything around it.

2. **Building mostly offline tests** — fake LLMs that return predetermined responses let you test the full stack without touching Ollama. 113 offline tests, 1 live test.

3. **Using `asyncio_mode = "auto"`** — eliminates boilerplate and makes async tests as easy to write as sync tests.

4. **Using `tmp_path` everywhere** — no test touches the real filesystem. No cleanup needed.

5. **Writing regression guards** — document bugs by testing the invariant that prevents them from recurring.

6. **Testing structure, not output** — `graph.get_graph().edges` tells you whether the graph is wired correctly without running it.

7. **Using `monkeypatch` for config** — safely override environment variables and settings attributes within test scope.

8. **Testing concurrency with timing** — generous upper bounds on elapsed time verify parallel execution without producing flaky tests.

9. **Using `capture_logs`** — structured logging is testable; make use of it.

10. **Marking live tests opt-in** — `addopts = "-m 'not live'"` keeps the fast test suite fast.

11. **Organizing by module** — one test file per source module makes tests easy to find and maintain.

The test suite is a first-class part of the codebase. A fast, reliable, comprehensive offline suite is what makes it possible to refactor confidently, ship frequently, and onboard new contributors without fear.

---

*Previous chapter: [Chapter 14 — The Code Review Workflow: Concurrent LLM Specialists](./14-code-review-workflow.md)*
