# Chapter 1 — Python Foundations for LLM Engineering

---

> *"Async Python is not a feature you add to a program. It is an architectural decision that shapes every layer of the system."*

---

## Overview

Before we can understand how saathi-langgraph works, we need to be fluent in the Python patterns it uses. This chapter is not a general Python tutorial — it targets specifically the patterns that appear throughout the LLM agent codebase and that are handled poorly by most introductory materials.

We cover fifteen topics in depth:

1. The asyncio model and why it dominates LLM engineering
2. `asyncio.gather` — true concurrency for I/O bound work
3. `asyncio.Semaphore` — bounding parallelism
4. `asyncio.create_subprocess_shell` — running shell commands asynchronously
5. Type annotations — the modern Python way
6. PEP 695 type parameters
7. Pydantic v2 — validation, settings, structured outputs
8. Dataclasses — when to use them instead of Pydantic
9. Context managers and `async with`
10. Exception handling patterns
11. Pathlib — working with filesystem paths
12. structlog — structured logging
13. f-strings and string formatting
14. List comprehensions and generators
15. Testing with pytest-asyncio

For every concept, we tie the theory directly to the saathi codebase. File references are given as `src/saathi/module.py`.

---

## 1. The asyncio Model

### 1.1 Why Async?

Every LLM call saathi makes is, at its core, an HTTP request. The model sends a JSON payload to Ollama's REST API (running at `http://localhost:11434`) and waits for a response. During that wait — which may last several seconds — the Python process could be doing other work.

This is the fundamental insight behind asyncio: when your program is waiting for **I/O** (network, disk, subprocess), the CPU is idle. A synchronous program wastes that idle time. An asynchronous program uses it.

For an agent that may execute five tool calls in a row, each of which is a file read, a subprocess, or another HTTP call, the difference between synchronous and asynchronous execution is not academic — it is the difference between a response time of fifteen seconds and three seconds.

### 1.2 The Event Loop Model

Python's asyncio is built around a single-threaded **event loop**. The event loop is a scheduler. It maintains a queue of coroutines that are ready to run and I/O operations that are waiting. When a coroutine hits an `await`, it suspends and returns control to the event loop, which can then run another coroutine. When the awaited I/O completes, the original coroutine is resumed.

```flow
Event Loop
│
├─ Run coroutine A until it awaits
│     A awaits network response
│
├─ Run coroutine B until it awaits
│     B awaits file read
│
├─ Network response arrives for A
│     Resume A with result
│
├─ File read completes for B
│     Resume B with result
```

There is no parallelism in the CPU sense — only one coroutine runs at a time. But there is **concurrency**: multiple coroutines make progress, interleaved, during the time that would otherwise be spent blocking.

### 1.3 Coroutines and `async def`

A coroutine is a function defined with `async def`. Calling it does not run the body — it returns a coroutine object. The body runs when you `await` the object or schedule it with the event loop.

```python
# This is a regular function — runs synchronously, blocks the thread
def fetch_sync(url: str) -> str:
    import urllib.request
    with urllib.request.urlopen(url) as f:
        return f.read().decode()

# This is a coroutine — can be awaited, does not block the event loop
async def fetch_async(url: str) -> str:
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text
```

When `fetch_async` hits `await client.get(url)`, it suspends. The event loop can run other coroutines while the HTTP response travels over the network.

### 1.4 The `await` Keyword

`await` can only appear inside an `async def` function. It takes an awaitable — a coroutine, a Task, or an object implementing `__await__` — and suspends the current coroutine until the awaitable completes.

```python
async def agent_node(state: AgentState) -> dict:
    # Suspend here while the LLM processes the request
    response = await llm.ainvoke(messages)
    return {"messages": [response]}
```

The `a` prefix in `ainvoke` is the LangChain convention for async methods. You will see it throughout: `ainvoke`, `astream`, `astream_events`.

### 1.5 Why LLM Calls Are Perfectly Suited for Async

An Ollama LLM call is, mechanically, this:

1. Serialize the prompt to JSON
2. Open an HTTP connection to `localhost:11434`
3. POST the JSON body
4. Wait for the response (this is the slow part)
5. Parse the response JSON
6. Return the model's text

Step 4 — waiting for the response — takes seconds. During those seconds, the Python process is doing nothing. Async lets you use that time productively: run other coroutines, handle other tool calls, stream intermediate results to the terminal.

```python
# Synchronous — tool calls happen one at a time
# Total time = sum of all tool times
def run_tools_sync(tool_calls):
    results = []
    for call in tool_calls:
        result = tool.invoke(call.args)  # blocks for each
        results.append(result)
    return results

# Asynchronous — tool calls happen concurrently
# Total time = max of all tool times
async def run_tools_async(tool_calls):
    coroutines = [tool.ainvoke(call.args) for call in tool_calls]
    results = await asyncio.gather(*coroutines)
    return results
```

If three tools each take 2 seconds, the synchronous version takes 6 seconds. The asynchronous version takes ~2 seconds.

### 1.6 Entering the Event Loop

In application code, you enter the event loop with `asyncio.run()`. Typer and Click apps that need async require a small wrapper:

```python
# src/saathi/cli.py
import asyncio
import typer

app = typer.Typer()

@app.command()
def main(
    model: str = typer.Option(None, "--model", "-m"),
    print_mode: bool = typer.Option(False, "--print", "-p"),
):
    asyncio.run(_async_main(model=model, print_mode=print_mode))

async def _async_main(model: str | None, print_mode: bool) -> None:
    # All async work happens here
    ...
```

The `asyncio.run()` call creates a new event loop, runs the coroutine until it completes, and closes the loop. Everything downstream can freely use `await`.

> **Note on running in Jupyter:** `asyncio.run()` fails inside Jupyter because a loop is already running. Use `await` directly or `nest_asyncio` to patch around this. saathi is a CLI tool and does not need to handle this case, but it is a common point of confusion.

---

## 2. `asyncio.gather` — True Concurrency for I/O

### 2.1 The Problem gather Solves

Suppose the LLM returns a response with five tool calls. A naive implementation processes them sequentially:

```python
results = []
for tool_call in tool_calls:
    result = await tool.ainvoke(tool_call.args)
    results.append(result)
```

This is correct but slow. Each `await` suspends the coroutine and waits for one tool to complete before starting the next. If each tool takes 1 second, this takes 5 seconds.

`asyncio.gather` solves this by running multiple coroutines **concurrently**:

```python
results = await asyncio.gather(
    tool_a.ainvoke(args_a),
    tool_b.ainvoke(args_b),
    tool_c.ainvoke(args_c),
)
```

All three coroutines are scheduled at once. Each runs until it awaits, then yields. The event loop interleaves them. All three complete in approximately the time of the slowest one.

### 2.2 The Signature

```python
asyncio.gather(*coros_or_futures, return_exceptions=False)
```

- Takes any number of awaitables (coroutines or Tasks)
- Returns a list of results in the same order as the inputs
- If `return_exceptions=True`, exceptions are returned as values rather than propagated
- If `return_exceptions=False` (the default), the first exception cancels remaining tasks and propagates

### 2.3 Timing Diagram

Consider four tool calls with varying durations:

```flow
Sequential (total: 9s):
Tool A [===2s===]
                  Tool B [=1s=]
                              Tool C [====3s====]
                                                  Tool D [=3s=]
Timeline: 0 ─── 2 ─── 3 ─────────── 6 ─────────── 9

Concurrent with gather (total: 3s):
Tool A [===2s===]
Tool B [=1s=]
Tool C [====3s====]
Tool D [=3s=]
Timeline: 0 ─────────── 3
```

The concurrent version runs in 3 seconds — the duration of the slowest tool. The sequential version runs in 9 seconds.

### 2.4 The saathi Pattern — Gathering Tool Calls

The actual implementation in `src/saathi/agent/tool_node.py` looks like this:

```python
async def hooked_tool_node(state: AgentState) -> dict:
    tool_calls = state["messages"][-1].tool_calls
    
    async def _guarded(call: ToolCall) -> ToolMessage:
        async with semaphore:
            # ... per-call pipeline
            result = await tools_by_name[call["name"]].ainvoke(call["args"])
            return ToolMessage(content=str(result), tool_call_id=call["id"])
    
    # Run all tool calls concurrently
    tool_messages = await asyncio.gather(
        *(_guarded(call) for call in tool_calls)
    )
    return {"messages": list(tool_messages)}
```

The generator expression `*(_guarded(call) for call in tool_calls)` creates one coroutine per tool call. `asyncio.gather` schedules all of them at once. The results come back in the same order as the tool calls — matching the IDs is important for LangChain's message threading.

### 2.5 Dynamic Coroutine Lists

Often you build the list of coroutines programmatically:

```python
# Build coroutine list dynamically
coroutines = [tool.ainvoke(call.args) for call in tool_calls]
results = await asyncio.gather(*coroutines)
```

The `*` unpacking works here because `gather` accepts `*args`. This is the most common real-world pattern: gather from a list, not a fixed set of arguments.

### 2.6 Error Handling with gather

By default, if any coroutine raises an exception, `gather` cancels the remaining coroutines and propagates the exception. This is usually the right behavior — if a critical tool fails, you want to know immediately.

For cases where you want to continue even if some tools fail, use `return_exceptions=True`:

```python
results = await asyncio.gather(*coroutines, return_exceptions=True)

for call, result in zip(tool_calls, results):
    if isinstance(result, Exception):
        # Handle failure for this specific tool call
        tool_messages.append(ToolMessage(
            content=f"Error: {result}",
            tool_call_id=call["id"]
        ))
    else:
        tool_messages.append(ToolMessage(
            content=str(result),
            tool_call_id=call["id"]
        ))
```

saathi's `hooked_tool_node` uses `return_exceptions=True` internally and wraps each call in a try/except that produces an error `ToolMessage` rather than crashing — this ensures the agent always gets a response for every tool call, even if some fail.

### 2.7 gather vs TaskGroup

Python 3.11 introduced `asyncio.TaskGroup` as a structured alternative to `gather`:

```python
# TaskGroup — Python 3.11+
async with asyncio.TaskGroup() as tg:
    tasks = [tg.create_task(coro) for coro in coroutines]
results = [task.result() for task in tasks]
```

`TaskGroup` has stricter cancellation semantics (all tasks cancel if any fails) and better error messages. For the parallel tool execution pattern, `gather` with `return_exceptions=True` is currently more natural because it maps directly to a list of results. Both are valid; the book uses `gather` because it matches the actual saathi code.

---

## 3. `asyncio.Semaphore` — Bounding Parallelism

### 3.1 The Problem

`asyncio.gather` with no limit would run all tool calls simultaneously, regardless of how many there are. If the LLM returns a response with 20 tool calls (unusual but possible for broad codebase searches), running 20 concurrent shell commands or file reads simultaneously could:

- Exhaust the Ollama connection pool
- Overwhelm the local filesystem with concurrent reads
- Consume excessive memory if each tool loads large files
- Cause subtle ordering bugs in tools with side effects

A **semaphore** is a counter that limits how many coroutines can run a given block of code simultaneously.

### 3.2 How Semaphore Works

```python
semaphore = asyncio.Semaphore(3)  # Allow at most 3 concurrent acquisitions

async def worker(id: int):
    async with semaphore:
        print(f"Worker {id} running")
        await asyncio.sleep(1)  # Simulate work
        print(f"Worker {id} done")

# Start 10 workers — at most 3 run at once
await asyncio.gather(*[worker(i) for i in range(10)])
```

When more than 3 workers try to acquire the semaphore simultaneously, the excess workers wait. As each worker finishes and releases, a waiting worker acquires and starts.

### 3.3 The `async with` Pattern

`asyncio.Semaphore` is an async context manager. The `async with semaphore:` pattern acquires on entry and releases on exit, even if an exception is raised:

```python
async with semaphore:
    # Semaphore acquired
    result = await do_work()
    # Semaphore released here, even if do_work() raises
```

This is equivalent to:

```python
await semaphore.acquire()
try:
    result = await do_work()
finally:
    semaphore.release()
```

The `async with` form is always preferred — it is impossible to forget the release.

### 3.4 The saathi Pattern — max_parallel_tools

In `src/saathi/agent/tool_node.py`, the semaphore is created from config:

```python
from saathi.config import Settings

def make_hooked_tool_node(tools: list, settings: Settings, hook_runner: HookRunner):
    tools_by_name = {tool.name: tool for tool in tools}
    semaphore = asyncio.Semaphore(settings.max_parallel_tools)
    
    async def hooked_tool_node(state: AgentState) -> dict:
        async def _guarded(call: ToolCall) -> ToolMessage:
            async with semaphore:
                # At most max_parallel_tools calls run simultaneously
                return await _execute_call(call, tools_by_name, hook_runner)
        
        messages = await asyncio.gather(
            *(_guarded(call) for call in state["messages"][-1].tool_calls),
            return_exceptions=True,
        )
        ...
    
    return hooked_tool_node
```

The `settings.max_parallel_tools` default is `8` — enough to run typical multi-file searches in parallel without overwhelming the system. Users can tune this via `SAATHI_MAX_PARALLEL_TOOLS`.

### 3.5 Semaphore as a Rate Limiter

The same pattern works for rate-limiting API calls:

```python
# Rate-limit to 10 concurrent requests to an external API
api_semaphore = asyncio.Semaphore(10)

async def call_api(item):
    async with api_semaphore:
        return await httpx_client.post("/api/process", json=item)

results = await asyncio.gather(*[call_api(item) for item in items])
```

This is useful when calling cloud APIs with rate limits, or when a downstream service has limited connection pool capacity.

### 3.6 Semaphore Scope

The semaphore must be created in the same event loop as the coroutines that use it. Creating it at module level can cause issues in tests that create new event loops. The saathi pattern creates the semaphore inside the factory function `make_hooked_tool_node`, which is called inside the running loop. Each call to `make_hooked_tool_node` (for instance, when the user changes models with `/model`) creates a fresh semaphore.

---

## 4. `asyncio.create_subprocess_shell` — Async Subprocesses

### 4.1 The Problem with subprocess

The standard library `subprocess.run()` blocks the thread:

```python
# This blocks for the entire duration of the command
result = subprocess.run(["git", "status"], capture_output=True, text=True)
```

In an async program, this is a bug. While `subprocess.run()` waits for `git status` to complete, the entire event loop is frozen. No other coroutines can run. If the subprocess takes 5 seconds, the agent is unresponsive for 5 seconds.

### 4.2 The Async Alternative

`asyncio.create_subprocess_shell` creates a subprocess and returns an awaitable. The event loop can run other coroutines while the subprocess executes:

```python
import asyncio

async def run_command(cmd: str) -> tuple[str, str, int]:
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return (
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
        proc.returncode,
    )
```

The `proc.communicate()` call reads stdout and stderr until the process exits, without blocking the event loop.

### 4.3 The saathi Hooks Runner Pattern

`src/saathi/hooks/runner.py` uses this to run user-defined shell hooks:

```python
async def run(self, event: str, env_vars: dict[str, str]) -> HookResult:
    """Run a hook command, returning stdout/stderr and return code."""
    cmd = self._resolve_command(event)
    if cmd is None:
        return HookResult(stdout="", stderr="", returncode=0)
    
    env = {**os.environ, **_build_env(env_vars)}
    
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        proc.kill()
        return HookResult(stdout="", stderr="Hook timed out", returncode=-1)
    
    return HookResult(
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
        returncode=proc.returncode,
    )
```

Key points:

- `asyncio.wait_for(proc.communicate(), timeout=60.0)` — hooks must complete within 60 seconds
- If they time out, the process is killed with `proc.kill()`
- `errors="replace"` in `.decode()` handles non-UTF-8 output gracefully

### 4.4 Shell vs Exec

`create_subprocess_shell` runs the command through the shell (`/bin/sh -c cmd`), which means:

- Shell features work: pipes, redirects, globs, `&&`, `||`
- Security risk: shell injection if `cmd` contains user-supplied strings
- Convenience: no need to split the command into a list

`create_subprocess_exec` runs the command directly:

```python
proc = await asyncio.create_subprocess_exec(
    "git", "status", "--porcelain",
    stdout=asyncio.subprocess.PIPE,
)
```

Use `exec` when you know the exact command and arguments at call time, and `shell` when you need shell features or when the command comes from a configuration file where the user wants shell power.

saathi's hooks use `shell` because the hook commands are user-defined and may use shell features. The built-in git tools use `exec` for safety.

### 4.5 Environment Variables for Hooks

The hook runner constructs an environment for each hook invocation:

```python
def _build_env(event: str, tool_name: str, tool_args: dict) -> dict[str, str]:
    return {
        "SAATHI_EVENT": event,
        "SAATHI_TOOL_NAME": tool_name,
        "SAATHI_TOOL_ARGS": json.dumps(tool_args),
        "SAATHI_TOOL_ARG_PATH": tool_args.get("path", ""),
    }
```

These environment variables allow hook scripts to know what triggered them:

```bash
#!/bin/bash
# .saathi/hooks/post_tool.sh
if [ "$SAATHI_TOOL_NAME" = "write_file" ]; then
    echo "File written: $SAATHI_TOOL_ARG_PATH"
    # Run linter on the written file
    ruff check "$SAATHI_TOOL_ARG_PATH"
fi
```

---

## 5. Type Annotations — The Modern Python Way

### 5.1 Why Type Annotations Matter

Python's type annotations are optional — the interpreter ignores them at runtime (with one key exception: Pydantic uses them). But they serve three critical purposes in an LLM engineering codebase:

**Static analysis:** mypy and pyright catch entire classes of bugs before runtime. Passing a `str` where a `list[BaseMessage]` is expected fails at type-check time, not during an agent turn at 11pm.

**Documentation:** Annotations are the most reliable form of documentation — they are read by tools and stay in sync with the code in a way that comments do not.

**LLM consumption:** When saathi reads its own codebase to help you, type annotations tell it what each function expects and returns. A well-annotated function is self-describing in a way that benefits both human readers and AI tools.

### 5.2 Basic Annotations

```python
# Variable annotations
name: str = "saathi"
count: int = 0
enabled: bool = True
items: list[str] = []

# Function annotations
def greet(name: str, times: int = 1) -> str:
    return (f"Hello, {name}!\n" * times).strip()

# None returns
def configure_logging(level: str) -> None:
    ...
```

### 5.3 Union Types — `str | None` (Python 3.10+)

Before Python 3.10, optional values required `Optional[str]` from `typing`. Since 3.10, the `|` operator works directly:

```python
# Old style
from typing import Optional
def get_model(name: Optional[str] = None) -> str: ...

# New style (Python 3.10+)
def get_model(name: str | None = None) -> str: ...
```

saathi uses the new style throughout. The `str | None` form is more readable and is the style mypy and pyright prefer.

For multiple types: `str | int | None` — a value that could be a string, integer, or None.

### 5.4 `TypedDict` — Typed Dictionaries

`TypedDict` creates a type for dictionaries with known keys and value types. LangGraph uses this extensively because `AgentState` must be a dict-like structure that LangGraph can serialize and deserialize.

```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# src/saathi/agent/state.py
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context_paths: list[str]
    mode: str   # "default" | "explain" | "refactor" | "debug"
    session_id: str
```

`TypedDict` is not a class in the normal sense — it does not enforce types at runtime. It is a hint to static type checkers and a form of structured documentation.

### 5.5 `Annotated` — Metadata on Types

`Annotated[T, metadata]` attaches metadata to a type hint. The metadata can be anything; it is ignored by the interpreter but used by frameworks that introspect types.

LangGraph's `add_messages` annotation is the key pattern in saathi:

```python
from typing import Annotated
from langgraph.graph.message import add_messages

messages: Annotated[list[BaseMessage], add_messages]
```

This tells LangGraph: "when a node returns `{"messages": [...]}`, do not replace the existing messages — call `add_messages(existing, new)` to merge them." This is the **reducer** pattern. Without this annotation, each node would overwrite the message list; with it, messages accumulate.

### 5.6 `dict[str, Any]` and `list[str]`

Since Python 3.9, built-in collection types support generic parameters directly:

```python
# Python 3.9+
config: dict[str, str] = {}
tool_args: dict[str, Any] = {}
paths: list[str] = []
results: list[tuple[str, int]] = []

# Old style (still valid, required for 3.8-)
from typing import Dict, List, Tuple
config: Dict[str, str] = {}
```

saathi uses the lowercase forms throughout. The `Any` type from `typing` is used sparingly — it disables type checking for that value, which is sometimes necessary for dynamic LLM outputs but should not be overused.

### 5.7 `Callable` — Function Types

When passing functions as arguments (as in the retry utility), use `Callable`:

```python
from collections.abc import Callable, Awaitable

# A function that takes no arguments and returns an awaitable int
fn: Callable[[], Awaitable[int]]

# A function that takes str and int, returns str
fn: Callable[[str, int], str]
```

`collections.abc.Callable` is the modern form (Python 3.9+). `typing.Callable` still works but is deprecated.

### 5.8 Type Narrowing

Static type checkers perform "type narrowing" — they refine a union type inside a conditional:

```python
def process(value: str | None) -> str:
    if value is None:
        return "default"
    # Here, type checker knows value: str (not str | None)
    return value.upper()
```

This is important in error handling code where you check for `isinstance`:

```python
results = await asyncio.gather(*coroutines, return_exceptions=True)
for result in results:
    if isinstance(result, Exception):
        # result: Exception here
        handle_error(result)
    else:
        # result: T (the coroutine return type) here
        process_result(result)
```

---

## 6. PEP 695 Type Parameters (Python 3.12+)

### 6.1 The Old Generic Syntax

Before Python 3.12, writing a generic function required importing `TypeVar` and using it in a signature:

```python
from typing import TypeVar, Callable, Awaitable

T = TypeVar("T")

async def retry_async(
    fn: Callable[[], Awaitable[T]],
    retries: int = 3,
    base_delay: float = 1.0,
) -> T:
    ...
```

The `TypeVar` declaration is verbose and the connection between `T` in the signature and the `TypeVar` object is not visually obvious — especially in a long file where `T = TypeVar("T")` appears far from the function that uses it.

### 6.2 PEP 695 Syntax

Python 3.12 introduced a new syntax for type parameters that is cleaner and more local:

```python
async def retry_async[T](
    fn: Callable[[], Awaitable[T]],
    retries: int = 3,
    base_delay: float = 1.0,
) -> T:
    ...
```

The `[T]` immediately after the function name declares `T` as a type parameter scoped to this function. No import needed, no separate declaration, no ambiguity about scope.

### 6.3 The saathi retry.py Implementation

Here is the complete retry function from `src/saathi/retry.py`:

```python
# src/saathi/retry.py
import asyncio
import logging
from collections.abc import Awaitable, Callable

import httpx

log = logging.getLogger(__name__)

# Exceptions that indicate a transient connection problem worth retrying
RETRYABLE = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    ConnectionError,
)


async def retry_async[T](
    fn: Callable[[], Awaitable[T]],
    *,
    retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Retry *fn* on transient connection errors with exponential back-off.

    Args:
        fn: Zero-argument async callable to retry.
        retries: Maximum number of attempts (including the first).
        base_delay: Initial sleep in seconds before the second attempt.
        max_delay: Cap for exponential back-off.
        sleep: Injectable sleep function (override in tests for speed).
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return await fn()
        except RETRYABLE as exc:
            last_exc = exc
            if attempt == retries:
                break
            delay = min(base_delay * 2 ** (attempt - 1), max_delay)
            log.warning(
                "retryable error, will retry",
                attempt=attempt,
                retries=retries,
                delay=delay,
                error=str(exc),
            )
            await sleep(delay)
    assert last_exc is not None
    raise last_exc
```

The `[T]` syntax in `async def retry_async[T](...)` says: "this function works with any return type T; the return type of `retry_async` is the same T as the return type of the `fn` callable."

This allows the type checker to infer the correct return type at the call site:

```python
# The type checker knows response: BaseMessage, not Any
response = await retry_async(lambda: llm.ainvoke(messages))
```

Without the generic, the return type would be `Any` and the type checker would lose information.

### 6.4 The Injectable Sleep Pattern

Notice `sleep: Callable[[float], Awaitable[None]] = asyncio.sleep`. This is a **dependency injection** pattern for testability.

`asyncio.sleep(1.0)` in a test makes the test take 1 second. By making `sleep` injectable, tests can pass a mock sleep:

```python
# tests/test_retry.py
async def test_retry_respects_delay_progression():
    delays: list[float] = []
    
    async def mock_sleep(seconds: float) -> None:
        delays.append(seconds)
    
    call_count = 0
    async def failing_fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise httpx.ConnectError("connection refused")
        return "success"
    
    result = await retry_async(failing_fn, retries=3, base_delay=1.0, sleep=mock_sleep)
    
    assert result == "success"
    assert delays == [1.0, 2.0]  # Exponential backoff: 1.0, 2.0
```

The test runs instantly because `mock_sleep` does nothing. Without the injectable parameter, you would need `unittest.mock.patch("asyncio.sleep", ...)`, which is less explicit.

### 6.5 Bounded Type Parameters

PEP 695 also supports type bounds:

```python
# T must be a subtype of BaseModel
def validate_many[T: BaseModel](items: list[dict], model: type[T]) -> list[T]:
    return [model.model_validate(item) for item in items]
```

And multiple type parameters:

```python
async def map_concurrent[T, U](
    items: list[T],
    fn: Callable[[T], Awaitable[U]],
) -> list[U]:
    return list(await asyncio.gather(*[fn(item) for item in items]))
```

---

## 7. Pydantic v2

### 7.1 What Pydantic Does

Pydantic is a data validation library. You define a model class with annotated fields, and Pydantic enforces that instances of that class have valid data:

```python
from pydantic import BaseModel

class Finding(BaseModel):
    file: str
    line: int | None
    severity: str  # "high" | "medium" | "low"
    message: str
    confidence: int  # 0-100
```

When you create a `Finding`, Pydantic validates the data:

```python
# Valid
finding = Finding(file="src/main.py", line=42, severity="high", message="...", confidence=85)

# Raises ValidationError — line must be int or None, not "forty-two"
finding = Finding(file="src/main.py", line="forty-two", severity="high", message="...", confidence=85)
```

### 7.2 The v2 API

Pydantic v2 (released 2023) changed several APIs. The key differences from v1:

| v1 | v2 |
| --- | --- |
| `@validator("field")` | `@field_validator("field")` |
| `class Config: ...` | `model_config = ConfigDict(...)` |
| `.dict()` | `.model_dump()` |
| `.json()` | `.model_dump_json()` |
| `parse_obj(data)` | `model_validate(data)` |
| `from pydantic import BaseSettings` | `from pydantic_settings import BaseSettings` |

### 7.3 Field Validators

```python
from pydantic import BaseModel, field_validator

class Finding(BaseModel):
    severity: str
    confidence: int
    
    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        allowed = {"high", "medium", "low", "info"}
        if v not in allowed:
            raise ValueError(f"severity must be one of {allowed}, got {v!r}")
        return v
    
    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: int) -> int:
        if not 0 <= v <= 100:
            raise ValueError(f"confidence must be 0-100, got {v}")
        return v
```

The `@classmethod` decorator is required in v2. The validator receives the value after basic type coercion and should either return the (possibly modified) value or raise `ValueError`.

### 7.4 Model Validators

`@model_validator` runs after all field validators and has access to the full model data:

```python
from pydantic import BaseModel, model_validator

class DateRange(BaseModel):
    start: str
    end: str
    
    @model_validator(mode="after")
    def check_range(self) -> "DateRange":
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self
```

### 7.5 The saathi Settings — pydantic-settings

`pydantic-settings` extends Pydantic's `BaseModel` with environment variable loading. `src/saathi/config.py`:

```python
# src/saathi/config.py
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SAATHI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_model: str = "gemma4:12b"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    context_window: int = 32_768
    max_tokens: int = 4_096
    max_parallel_tools: int = 8
    review_min_confidence: int = 70

    @computed_field
    @property
    def history_token_budget(self) -> int:
        """75% of context window reserved for history."""
        return int(self.context_window * 0.75)
```

Key points:

**`env_prefix = "SAATHI_"`:** Environment variables are read with this prefix stripped. `SAATHI_OLLAMA_MODEL` maps to `ollama_model`. This prevents namespace collisions with other applications.

**`env_file = ".env"`:** Pydantic-settings will read a `.env` file in the current directory if it exists. Variables in the environment always take precedence over `.env` values.

**`extra = "ignore"`:** Unknown environment variables (like `SAATHI_FUTURE_OPTION`) are silently ignored rather than causing a validation error. This makes the settings forward-compatible.

**`@computed_field`:** A computed field is a property that appears in `.model_dump()` output and can be used like a regular field, but is derived from other fields. `history_token_budget` is always 75% of `context_window` — users should not set it directly.

### 7.6 Using Settings in the Application

The recommended pattern is to create a single `Settings()` instance per process and pass it around:

```python
# At CLI startup
settings = Settings()

# Pass to graph builder
graph = build_graph(settings=settings)
```

This approach is testable — tests can create `Settings(ollama_model="test-model")` or override specific fields. It is also explicit — the dependency on configuration is visible in function signatures.

Avoid calling `Settings()` inside library functions. A function that calls `Settings()` internally is hard to test and creates an implicit global dependency.

### 7.7 Structured LLM Outputs with Pydantic

One of the most powerful uses of Pydantic in LLM engineering is structured output validation. LangChain's `with_structured_output` method uses a Pydantic model to force the LLM to return JSON matching the schema:

```python
from pydantic import BaseModel

class CodeReviewFindings(BaseModel):
    findings: list[Finding]
    summary: str
    overall_quality: str

# LLM will return JSON parseable as CodeReviewFindings
structured_llm = llm.with_structured_output(CodeReviewFindings)
result: CodeReviewFindings = await structured_llm.ainvoke(prompt)
```

saathi's `src/saathi/review.py` uses this pattern to extract structured code review findings from the LLM's analysis.

---

## 8. Dataclasses — When to Use Them

### 8.1 dataclass vs BaseModel

Pydantic `BaseModel` and Python `@dataclass` look similar but serve different purposes:

| | `@dataclass` | `BaseModel` |
| --- | --- | --- |
| Runtime validation | No | Yes |
| Serialization | Basic (`__repr__`, `asdict`) | Rich (`.model_dump()`, `.model_dump_json()`) |
| JSON schema | No | Yes (`.model_json_schema()`) |
| From dict | No | `.model_validate(dict)` |
| Performance | Faster construction | Slower (validation overhead) |
| Use case | Internal data structures | API boundaries, config, LLM outputs |

Use `@dataclass` for internal data structures where you control construction and do not need serialization or validation. Use `BaseModel` when data crosses an API boundary (LLM output, config files, HTTP responses).

### 8.2 The HookConfig Dataclass

`src/saathi/hooks/runner.py` uses dataclasses for the hook configuration:

```python
from dataclasses import dataclass, field

@dataclass
class HookConfig:
    """Configuration loaded from .saathi/hooks.json."""
    pre_tool: dict[str, str] = field(default_factory=dict)
    post_tool: dict[str, str] = field(default_factory=dict)
    post_turn: str | None = None
    block_paths: list[str] = field(default_factory=list)
```

`HookConfig` is constructed from parsed JSON in `load_hook_config()`. There is no external validation needed (the code handles malformed config gracefully), and it is never serialized back to JSON. A dataclass is the right choice.

The `field(default_factory=dict)` pattern is important. This is equivalent to:

```python
# WRONG — mutable default shared across all instances
@dataclass
class Bad:
    items: list = []  # All instances share the same list object!

# RIGHT — fresh list for each instance
@dataclass 
class Good:
    items: list = field(default_factory=list)
```

### 8.3 frozen=True for Immutable Data

```python
@dataclass(frozen=True)
class ToolCallRecord:
    """Immutable record of a tool call for audit logging."""
    tool_name: str
    args: str  # JSON-serialized
    result_preview: str
    duration_ms: int
    timestamp: str
```

`frozen=True` makes the dataclass immutable (setting attributes raises `FrozenInstanceError`) and hashable. This is useful for audit records, configuration objects that should not change after creation, and any data that should be treated as a value.

### 8.4 `__post_init__`

Dataclasses support a `__post_init__` method that runs after `__init__`:

```python
@dataclass
class HookRunner:
    config: HookConfig
    project_root: Path
    
    def __post_init__(self) -> None:
        # Validate that block_paths are valid patterns
        for pattern in self.config.block_paths:
            if not isinstance(pattern, str):
                raise ValueError(f"block_paths must be strings, got {type(pattern)}")
```

This is a lightweight alternative to Pydantic validators for dataclasses.

---

## 9. Context Managers and `async with`

### 9.1 The Context Manager Protocol

A context manager is any object implementing `__enter__` / `__exit__` (sync) or `__aenter__` / `__aexit__` (async). The `with` statement guarantees that `__exit__` is called even if an exception occurs inside the block:

```python
# File is always closed, even if processing raises
with open("file.txt") as f:
    data = f.read()
    process(data)  # Exception here? File still closes.
```

### 9.2 `async with` for Async Resources

The async equivalent uses `__aenter__` and `__aexit__`:

```python
# AsyncClient is properly closed even if requests raise
async with httpx.AsyncClient() as client:
    response = await client.get(url)
```

For aiosqlite connections, which are async context managers:

```python
async with aiosqlite.connect("db.sqlite3") as conn:
    await conn.execute("CREATE TABLE IF NOT EXISTS ...")
    await conn.commit()
# Connection closed here
```

### 9.3 The saathi Checkpoint Connection Pattern

LangGraph's `AsyncSqliteSaver` normally uses a context manager:

```python
# Normal usage — connection opened and closed automatically
async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    # graph is only valid inside this block!
```

This pattern is awkward for a long-running REPL where the connection must persist for the entire session. saathi's `src/saathi/agent/graph.py` manages the connection explicitly:

```python
async def build_graph(settings: Settings, db_path: Path) -> tuple[CompiledGraph, Connection]:
    """Build the compiled graph with an explicit aiosqlite connection.
    
    Returns the graph and connection; caller is responsible for closing.
    """
    conn = await aiosqlite.connect(str(db_path))
    checkpointer = AsyncSqliteSaver(conn)
    
    # ... build and compile graph ...
    
    return graph, conn

async def close_graph(conn: Connection) -> None:
    """Close the checkpoint database connection."""
    await conn.close()
```

In `cli.py`, the session manages the lifecycle:

```python
graph, conn = await build_graph(settings, db_path)
try:
    await _interactive_session(graph, settings)
finally:
    await close_graph(conn)
```

The `try/finally` guarantees the connection closes even if the session raises.

### 9.4 `@contextmanager` and `@asynccontextmanager`

You can write context managers as generators using the decorator:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def timer(label: str):
    import time
    start = time.monotonic()
    try:
        yield  # Code inside the `async with` block runs here
    finally:
        elapsed = time.monotonic() - start
        print(f"{label}: {elapsed:.3f}s")

async def main():
    async with timer("tool execution"):
        result = await run_tool(args)
```

This pattern is used in tests to measure performance and in the agent for optional timing instrumentation.

---

## 10. Exception Handling Patterns

### 10.1 The Core Principle: Only Catch What You Can Handle

The most important rule in exception handling:

> **Catch exceptions at the point where you have enough context to handle them meaningfully. Everywhere else, let them propagate.**

A function deep in the call stack that catches all exceptions with `except Exception` and logs them does two harmful things: it hides bugs that should surface, and it leaves the system in an unknown state.

```python
# BAD — swallows all exceptions, hides bugs
async def call_tool(args: dict) -> str:
    try:
        return await tool.ainvoke(args)
    except Exception as e:
        log.error("tool failed", error=e)
        return ""  # Caller has no idea anything went wrong

# BETTER — only catch what you can handle; let others propagate
async def call_tool(args: dict) -> str:
    try:
        return await tool.ainvoke(args)
    except httpx.ConnectError:
        # We know this is a connection problem; we can return an informative message
        return "Error: Could not connect to Ollama. Is it running?"
    # Other exceptions propagate to the caller
```

### 10.2 Retryable vs Non-Retryable Errors

Not all exceptions should be retried. Retrying a `400 Bad Request` (malformed prompt) is pointless — the request will fail the same way every time. Retrying a `ConnectionError` (Ollama restarted) makes sense — the connection may succeed after a brief delay.

saathi's `src/saathi/retry.py` encodes this as the `RETRYABLE` tuple:

```python
RETRYABLE = (
    httpx.ConnectError,    # Could not establish connection
    httpx.ConnectTimeout,  # Connection establishment timed out
    httpx.PoolTimeout,     # No available connections in pool
    ConnectionError,       # Python stdlib connection error
)
```

Only these exception types are retried. An `httpx.HTTPStatusError` (4xx, 5xx response) is not retried — Ollama responded, just with an error. A `json.JSONDecodeError` is not retried — the model returned malformed output; waiting will not help.

### 10.3 The Exception Hierarchy

Understanding the exception hierarchy helps write precise `except` clauses:

```flow
BaseException
├── SystemExit
├── KeyboardInterrupt
├── GeneratorExit
└── Exception
    ├── ValueError
    ├── TypeError
    ├── OSError
    │   ├── FileNotFoundError
    │   ├── PermissionError
    │   └── ConnectionError
    │       ├── ConnectionRefusedError
    │       └── ConnectionResetError
    └── httpx.HTTPError
        ├── httpx.ConnectError
        ├── httpx.ConnectTimeout
        └── httpx.HTTPStatusError
```

Catching `OSError` catches all I/O errors including file errors and connection errors. Catching `ConnectionError` is more precise — only network connection failures. Always prefer the most specific exception type that correctly describes what you are handling.

### 10.4 Tool Error Handling Pattern

Tools should return informative error strings rather than raising exceptions, because the LLM needs to see the error to reason about it:

```python
# tools/filesystem.py
async def read_file(path: str) -> str:
    """Read a file and return its contents."""
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
        return content
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except OSError as e:
        return f"Error reading {path}: {e}"
```

By returning error strings, the tool produces a `ToolMessage` that the LLM can read and respond to ("The file was not found. Let me check the directory listing to find the correct path."). If the tool raised an exception instead, the tool_node would catch it and produce a generic error message — less informative.

### 10.5 `except*` — Python 3.11 Exception Groups

Python 3.11 introduced `ExceptionGroup` and `except*` for handling multiple exceptions from concurrent operations:

```python
# When multiple concurrent operations fail, Python 3.11 can collect them
try:
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(op()) for op in operations]
except* ValueError as eg:
    # Handle all ValueErrors from any task
    for exc in eg.exceptions:
        log.error("validation failed", error=exc)
except* httpx.ConnectError as eg:
    # Handle all connection errors
    ...
```

saathi currently uses `asyncio.gather(return_exceptions=True)` which handles errors inline rather than using exception groups. `except*` is shown here for completeness — it is the modern pattern for structured concurrency error handling.

---

## 11. Pathlib

### 11.1 The Case Against String Paths

String paths are fragile:

```python
# Fragile — OS-dependent separator, no validation
path = home_dir + "/" + ".saathi" + "/" + "memory.json"

# Better but still fragile — what if home_dir ends with "/"?
path = os.path.join(home_dir, ".saathi", "memory.json")

# Robust — Path handles all of this correctly
path = Path(home_dir) / ".saathi" / "memory.json"
```

`pathlib.Path` objects:

- Work on all platforms (Windows uses `\`, Unix uses `/`, but `Path` handles both)
- Validate that the path is syntactically valid
- Provide methods that are cleaner than `os.path.*`
- Are immutable and hashable

### 11.2 Common Path Operations

```python
from pathlib import Path

# Create paths
home = Path.home()              # /home/user or C:\Users\user
cwd = Path.cwd()                # Current working directory
p = Path("/some/path/file.txt")

# Navigation
parent = p.parent               # /some/path
name = p.name                   # file.txt
stem = p.stem                   # file
suffix = p.suffix               # .txt
parts = p.parts                 # ('/', 'some', 'path', 'file.txt')

# Building paths
config = home / ".saathi" / "config.json"
child = p.parent / "other.txt"

# Testing
p.exists()                      # True if path exists
p.is_file()                     # True if it's a file
p.is_dir()                      # True if it's a directory

# Reading and writing
content = p.read_text(encoding="utf-8")
p.write_text("content", encoding="utf-8")
data = p.read_bytes()

# Directory operations
p.parent.mkdir(parents=True, exist_ok=True)
files = list(p.parent.glob("*.py"))
all_py = list(p.parent.rglob("*.py"))
```

### 11.3 The saathi Memory Path Pattern

`src/saathi/memory/store.py` uses pathlib to locate memory files:

```python
from pathlib import Path

def _global_memory_path() -> Path:
    return Path.home() / ".saathi" / "memory.json"

def _project_memory_path() -> Path:
    return Path.cwd() / ".saathi" / "memory.json"

class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
    
    def _load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))
    
    def save(self, key: str, value: str) -> None:
        data = self._load()
        data[key] = value
        self.path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8"
        )
```

`Path.home()` is cross-platform: it returns `~` expanded correctly on every OS. `mkdir(parents=True, exist_ok=True)` creates the entire directory tree without raising if it already exists.

### 11.4 Path in Tests — `tmp_path`

pytest's `tmp_path` fixture provides a temporary directory as a `Path` object:

```python
# tests/test_memory.py
async def test_memory_save_and_load(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.json")
    
    store.save("project", "saathi-langgraph")
    store.save("language", "python")
    
    assert store.get("project") == "saathi-langgraph"
    assert store.get("language") == "python"
    
    all_items = store.all()
    assert len(all_items) == 2
```

`tmp_path` is automatically cleaned up after the test. It is isolated per test run, preventing tests from interfering with each other or with real user data.

### 11.5 `expanduser()` for Paths with `~`

When accepting paths from users or config files:

```python
# A user might specify "~/projects/myapp" in config
raw_path = settings.context_path  # "~/projects/myapp"
resolved = Path(raw_path).expanduser().resolve()
# On Unix: /home/username/projects/myapp
# On Windows: C:\Users\username\projects\myapp
```

Always call `expanduser()` when accepting user-specified paths. `.resolve()` additionally resolves symlinks and makes the path absolute.

---

## 12. structlog — Structured Logging

### 12.1 The Problem with Standard Logging

Python's `logging` module produces text:

```text
2026-07-09 11:30:00,123 WARNING saathi.retry: retryable error, will retry attempt=1 retries=3 delay=1.0 error=connection refused
```

This is human-readable but machine-unfriendly. Parsing log lines to extract `attempt=1` requires fragile regex. When logs go to a log aggregation system (ELK, Splunk, CloudWatch), you want JSON:

```json
{"timestamp": "2026-07-09T11:30:00.123Z", "level": "warning", "logger": "saathi.retry", 
 "event": "retryable error, will retry", "attempt": 1, "retries": 3, "delay": 1.0, 
 "error": "connection refused"}
```

### 12.2 structlog Basics

structlog wraps standard logging with a structured API:

```python
import structlog

log = structlog.get_logger(__name__)

# Key-value pairs are first-class — not interpolated into a string
log.warning(
    "retryable error, will retry",
    attempt=attempt,
    retries=retries,
    delay=delay,
    error=str(exc),
)
```

In development, structlog renders this as colorful, readable console output. In production, it renders as JSON. The rendering format is configurable; the calling code stays the same.

### 12.3 The saathi configure_logging Pattern

`src/saathi/logging_config.py`:

```python
import logging
import sys
import structlog


def configure_logging(level: str = "WARNING", json_logs: bool = False) -> None:
    """Configure structlog for the application.
    
    Logs always go to stderr to keep stdout clean for --print mode.
    """
    log_level = getattr(logging, level.upper(), logging.WARNING)
    
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    
    if json_logs:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    
    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )
    
    # Also configure stdlib logging to go to stderr
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=log_level,
    )
```

The critical detail: `logger_factory=structlog.PrintLoggerFactory(file=sys.stderr)`. All log output goes to **stderr**, not stdout. This is essential for `--print` mode, where stdout must contain only the agent's response.

```bash
# In print mode, stdout is the response, stderr is logs
saathi --print "what is 2+2" > response.txt 2> debug.log
```

### 12.4 Why Not `print()`?

During development, `print()` seems simpler. But:

- `print()` goes to stdout, contaminating `--print` mode output
- `print()` has no levels — you cannot filter debug vs warning
- `print()` has no context — you cannot easily add request IDs, tool names, etc.
- `print()` has no structured data — log aggregation tools cannot parse it

The structlog call `log.debug("tool invoked", tool=name, args=args)` adds tool name and args as structured fields that can be filtered, aggregated, and searched. `print(f"tool invoked: {name} args={args}")` cannot be parsed reliably.

### 12.5 Bound Loggers — Adding Context

structlog supports binding context that is automatically included in all subsequent log calls from that logger:

```python
# Add session context once, gets included in all logs
session_log = log.bind(session_id=session_id, thread_id=thread_id)

session_log.info("session started")
session_log.debug("tool invoked", tool_name=tool_name)
# Both log entries include session_id and thread_id
```

saathi uses this to bind the session ID and thread ID to the logger at the start of each session, providing correlation across all log messages for that session.

---

## 13. f-strings and String Formatting

### 13.1 f-string Basics

f-strings (formatted string literals) embed expressions directly in string literals:

```python
name = "saathi"
version = "1.0"
message = f"Welcome to {name} v{version}!"
# "Welcome to saathi v1.0!"
```

Any Python expression works inside `{}`:

```python
count = 42
summary = f"Found {count} {'finding' if count == 1 else 'findings'}"
```

### 13.2 Format Specs

f-strings support format specifications after `:`:

```python
pi = 3.14159265358979
print(f"Pi is approximately {pi:.2f}")    # "Pi is approximately 3.14"
print(f"Pi is approximately {pi:.4f}")    # "Pi is approximately 3.1416"

duration_ms = 1234
print(f"Duration: {duration_ms:,}ms")    # "Duration: 1,234ms"

# Padding and alignment
label = "tokens"
count = 42
print(f"{label:>10}: {count:5d}")         # "    tokens:    42"
```

### 13.3 `!r` for Debug Output

The `!r` conversion calls `repr()` on the value — useful for distinguishing None from "None" and empty string from whitespace:

```python
value = None
print(f"value={value!r}")    # "value=None"

value = "None"
print(f"value={value!r}")    # "value='None'"

value = ""
print(f"value={value!r}")    # "value=''"

value = "  "
print(f"value={value!r}")    # "value='  '"
```

saathi uses `!r` in error messages to make empty strings and None values visible.

### 13.4 Multiline f-strings in Prompts

The system prompt in `src/saathi/agent/prompts.py` is built with multiline f-strings:

```python
def build_system_prompt(
    mode: str,
    project_instructions: str | None,
    memory_block: str | None,
    context_paths: list[str],
) -> str:
    parts: list[str] = [BASE_PROMPT]
    
    if project_instructions:
        parts.append(f"""
## Project Instructions

{project_instructions}
""")
    
    if memory_block:
        parts.append(f"""
## Memory

{memory_block}
""")
    
    if context_paths:
        path_list = "\n".join(f"- {p}" for p in context_paths)
        parts.append(f"""
## Context Files

The following files have been loaded as context:
{path_list}
""")
    
    if mode in _MODE_ADDENDA:
        parts.append(_MODE_ADDENDA[mode])
    
    return "\n".join(parts)
```

Indented triple-quoted f-strings are the clearest way to build structured text blocks. The dedent pattern (`textwrap.dedent(f"""...""")`) removes common leading whitespace when the string content needs to be left-aligned.

---

## 14. List Comprehensions and Generator Expressions

### 14.1 List Comprehensions

A list comprehension builds a new list by applying an expression to each element:

```python
# Imperative style
tool_names = []
for tool in tools:
    tool_names.append(tool.name)

# Comprehension style
tool_names = [tool.name for tool in tools]
```

With filtering:

```python
# Only paths that exist
valid_paths = [p for p in paths if Path(p).exists()]

# Tool names starting with "git_"
git_tools = [t.name for t in tools if t.name.startswith("git_")]
```

Nested comprehensions (for tables, matrices):

```python
# Flatten a list of lists
all_findings = [f for file_findings in results for f in file_findings]
```

### 14.2 Dict and Set Comprehensions

```python
# Build a tools_by_name dict
tools_by_name: dict[str, BaseTool] = {tool.name: tool for tool in tools}

# Build a set of unique tool names
tool_name_set: set[str] = {call["name"] for call in tool_calls}
```

The `tools_by_name` dict comprehension is used extensively in saathi's tool node to look up tools by name in O(1) time.

### 14.3 Generator Expressions

A generator expression is like a list comprehension but lazy — it produces values on demand without building the entire list in memory:

```python
# List comprehension — evaluates everything immediately, stores list in memory
coroutines = [tool.ainvoke(call.args) for call in tool_calls]

# Generator expression — creates a generator object, lazy evaluation
coroutines = (tool.ainvoke(call.args) for call in tool_calls)
```

For `asyncio.gather`, you can pass a generator expression with `*`:

```python
# These are equivalent
await asyncio.gather(*[_guarded(call) for call in tool_calls])
await asyncio.gather(*(_guarded(call) for call in tool_calls))
```

The generator form avoids building an intermediate list, which matters when `tool_calls` is large. For the typical case of 3-10 tool calls, the difference is negligible. saathi uses the generator form for stylistic consistency.

### 14.4 When Not to Use Comprehensions

Comprehensions are not always clearer. Multi-step transformations, complex conditions, or side effects belong in explicit loops:

```python
# Comprehension — too complex, hard to read
processed = [
    item.strip().lower().replace("-", "_")
    for item in raw_items
    if item and item.strip() and not item.startswith("#")
]

# Loop — clearer for complex logic
processed = []
for item in raw_items:
    if not item or not item.strip():
        continue
    if item.startswith("#"):  # Skip comments
        continue
    processed.append(item.strip().lower().replace("-", "_"))
```

The rule of thumb: if the comprehension does not fit comfortably on one or two lines, write a loop.

---

## 15. Testing with pytest-asyncio

### 15.1 The Problem with Async Tests

pytest runs test functions synchronously. An `async def test_*` function is a coroutine — calling it returns a coroutine object, not a result. Without extra support, pytest would see a coroutine object that is always truthy and pass every test.

pytest-asyncio solves this by wrapping async tests in an event loop. With `asyncio_mode = "auto"`, no decorators are needed — any `async def test_*` function runs in an event loop automatically.

### 15.2 pyproject.toml Configuration

From `saathi-langgraph/pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
pythonpath = ["src"]
markers = ["live: integration tests requiring Ollama (deselect with -m 'not live')"]
addopts = "-m 'not live'"
```

Key settings:

**`asyncio_mode = "auto"`:** All `async def test_*` functions automatically run in an event loop. No `@pytest.mark.asyncio` decorator needed.

**`pythonpath = ["src"]`:** Adds `src/` to the Python path, so `import saathi` works without installing the package in editable mode.

**`addopts = "-m 'not live'"`:** Excludes tests marked `@pytest.mark.live` by default. Live tests require Ollama running and are run separately with `pytest -m live`.

### 15.3 Writing Async Tests

```python
# tests/test_retry.py
import pytest
import httpx
from saathi.retry import retry_async, RETRYABLE


async def test_retry_succeeds_immediately() -> None:
    """A function that succeeds on first call needs no retries."""
    call_count = 0
    
    async def fn() -> str:
        nonlocal call_count
        call_count += 1
        return "success"
    
    result = await retry_async(fn, retries=3)
    
    assert result == "success"
    assert call_count == 1


async def test_retry_on_connect_error() -> None:
    """Connection errors are retried up to the limit."""
    call_count = 0
    
    async def fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise httpx.ConnectError("connection refused")
        return "success"
    
    # Use mock_sleep to make the test instant
    async def mock_sleep(_: float) -> None:
        pass
    
    result = await retry_async(fn, retries=3, sleep=mock_sleep)
    
    assert result == "success"
    assert call_count == 3


async def test_retry_raises_after_exhaustion() -> None:
    """After all retries are exhausted, the last exception propagates."""
    async def always_fails() -> str:
        raise httpx.ConnectError("connection refused")
    
    async def mock_sleep(_: float) -> None:
        pass
    
    with pytest.raises(httpx.ConnectError, match="connection refused"):
        await retry_async(always_fails, retries=3, sleep=mock_sleep)


async def test_non_retryable_error_propagates_immediately() -> None:
    """Errors not in RETRYABLE are not retried."""
    call_count = 0
    
    async def fn() -> str:
        nonlocal call_count
        call_count += 1
        raise ValueError("bad input")
    
    with pytest.raises(ValueError, match="bad input"):
        await retry_async(fn, retries=5)
    
    assert call_count == 1  # Called only once — no retries for ValueError
```

### 15.4 The `tmp_path` Fixture

pytest provides `tmp_path` as a `pathlib.Path` pointing to a temporary directory unique to each test:

```python
# tests/test_memory.py
from pathlib import Path
from saathi.memory.store import MemoryStore


async def test_memory_persistence(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    store = MemoryStore(path)
    
    store.save("key1", "value1")
    
    # Create a new store instance pointing to the same file
    store2 = MemoryStore(path)
    assert store2.get("key1") == "value1"


async def test_memory_delete(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.json")
    store.save("to_delete", "gone")
    store.save("to_keep", "here")
    
    store.delete("to_delete")
    
    assert store.get("to_delete") is None
    assert store.get("to_keep") == "here"
```

`tmp_path` is automatically cleaned up after the test suite finishes. Each test gets its own directory, so tests cannot interfere with each other.

### 15.5 The `monkeypatch` Fixture

`monkeypatch` temporarily replaces attributes, environment variables, or functions:

```python
# tests/test_config.py
from saathi.config import Settings


def test_settings_from_env(monkeypatch) -> None:
    """Settings should read from environment variables."""
    monkeypatch.setenv("SAATHI_OLLAMA_MODEL", "llama3.2:7b")
    monkeypatch.setenv("SAATHI_MAX_PARALLEL_TOOLS", "4")
    
    settings = Settings()
    
    assert settings.ollama_model == "llama3.2:7b"
    assert settings.max_parallel_tools == 4


def test_settings_defaults() -> None:
    """Default values should be stable."""
    settings = Settings()
    
    assert settings.temperature == 0.1
    assert settings.max_parallel_tools == 8
```

`monkeypatch.setenv` sets an environment variable for the duration of the test and restores the original value afterwards. It works with `pydantic-settings` because `Settings()` reads environment variables at construction time.

### 15.6 The `capfd` Fixture — Capturing Output

`capfd` captures file descriptor output (stdout and stderr), which is necessary for testing Rich console output:

```python
# tests/test_print_mode.py
def test_print_mode_writes_to_stdout(capfd) -> None:
    """--print mode should write the response to stdout."""
    # ... invoke CLI in print mode ...
    
    captured = capfd.readouterr()
    assert "2 + 2" in captured.out or "4" in captured.out
    # stderr may contain logs but stdout should be clean


def test_logging_goes_to_stderr(capfd) -> None:
    """Log output should never appear on stdout."""
    from saathi.logging_config import configure_logging
    import structlog
    
    configure_logging(level="DEBUG")
    log = structlog.get_logger("test")
    log.debug("test log message")
    
    captured = capfd.readouterr()
    assert "test log message" not in captured.out   # not on stdout
    assert "test log message" in captured.err        # on stderr
```

`capfd.readouterr()` returns a `(out, err)` namedtuple with the captured content and clears the buffers.

### 15.7 Mocking LLM Calls

Integration tests that call a real Ollama instance are marked `@pytest.mark.live`. Unit tests mock the LLM:

```python
# tests/test_graph.py
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage


async def test_agent_node_calls_llm(tmp_path) -> None:
    """agent_node should invoke the LLM and return its response."""
    mock_llm = AsyncMock()
    mock_response = AIMessage(content="The answer is 42.")
    mock_llm.ainvoke.return_value = mock_response
    
    from saathi.agent.nodes import make_agent_node
    node = make_agent_node(llm=mock_llm, memory_store=None)
    
    state = {
        "messages": [HumanMessage(content="What is the answer?")],
        "context_paths": [],
        "mode": "default",
        "session_id": "test-session",
    }
    
    result = await node(state)
    
    assert result["messages"] == [mock_response]
    mock_llm.ainvoke.assert_called_once()
```

`AsyncMock` is the async version of `MagicMock` — it supports `await mock.ainvoke(...)`.

### 15.8 Parallel Test Execution

The test for parallel tool execution (`tests/test_parallel.py`) uses timing to verify concurrency:

```python
import asyncio
import time


async def test_tools_run_in_parallel() -> None:
    """Multiple tool calls should execute concurrently, not sequentially."""
    call_times: list[float] = []
    
    async def slow_tool(delay: float) -> str:
        await asyncio.sleep(delay)
        call_times.append(time.monotonic())
        return f"done after {delay}s"
    
    start = time.monotonic()
    results = await asyncio.gather(
        slow_tool(0.1),
        slow_tool(0.1),
        slow_tool(0.1),
    )
    elapsed = time.monotonic() - start
    
    # Sequential would take 0.3s; parallel should take ~0.1s
    assert elapsed < 0.2, f"Expected parallel execution, took {elapsed:.2f}s"
    assert len(results) == 3
```

This test verifies the fundamental property that `asyncio.gather` provides: concurrent execution with total time bounded by the slowest coroutine, not the sum.

---

## Chapter Summary

We have covered fifteen Python patterns that form the technical foundation of the saathi codebase:

| Pattern | Key insight | File |
| --- | --- | --- |
| asyncio | LLM calls are I/O — `await` them | `agent/nodes.py` |
| `gather` | Run multiple tools concurrently | `agent/tool_node.py` |
| `Semaphore` | Bound concurrency to prevent overload | `agent/tool_node.py` |
| `create_subprocess_shell` | Run hooks without blocking the event loop | `hooks/runner.py` |
| Type annotations | Annotations are load-bearing documentation | `agent/state.py` |
| PEP 695 generics | Clean, local type parameters | `retry.py` |
| Pydantic v2 | Validate at API boundaries; configure with env vars | `config.py`, `review.py` |
| Dataclasses | Lightweight internal structures | `hooks/runner.py` |
| `async with` | Resource lifecycle management | `agent/graph.py` |
| Exception handling | Catch what you can handle; let the rest propagate | `retry.py`, `tools/` |
| Pathlib | Cross-platform path manipulation | `memory/store.py` |
| structlog | Structured logs to stderr, never stdout | `logging_config.py` |
| f-strings | Readable string construction | `agent/prompts.py` |
| Comprehensions | Idiomatic list and dict building | `agent/tool_node.py` |
| pytest-asyncio | Async tests that run without decorators | `tests/` |

Each pattern is not an isolated technique — they compose. The retry function uses PEP 695 generics and the injectable sleep pattern for testability. The tool node uses `gather` + `Semaphore` + `async with` together. The hooks runner uses `create_subprocess_shell` + `asyncio.wait_for` + dataclasses. Understanding each pattern individually prepares you to read the composed system in the chapters ahead.

---

**Next:** [Chapter 2 — LangGraph Core Concepts](./02-langgraph-core.md)
