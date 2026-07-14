# Chapter 7 — Async Python in FastAPI

> **What you'll learn:** what `async`/`await` actually means, how Python's event loop works, when to use `async def` vs `def`, what "blocking the event loop" means and why it matters, and how the Saathi API uses async throughout.

---

## The problem async solves

Imagine a web server that can handle one request at a time. A request comes in, calls a database, waits 50ms for the result, returns a response. While it's waiting, every other incoming request is stuck in a queue.

Now imagine 100 users hit the server simultaneously. User 1 gets served. Users 2–100 wait. The server is doing nothing useful during those 50ms waits — the CPU is idle, the thread is blocked.

Async I/O solves this: instead of blocking a thread while waiting for the database, you *suspend* the current task and let the event loop handle other requests. When the database responds, the suspended task is resumed.

This is not multithreading. There is still one thread. But instead of one thread blocking on I/O, one thread juggles many concurrent I/O operations.

---

## The event loop — a mental model

Python's async model is built around an **event loop** — a loop that:

1. Checks if any I/O operations have completed (database response arrived? HTTP response ready?)
2. Resumes the coroutines that were waiting on those operations
3. Starts running them until they hit another `await` (another I/O wait)
4. Goes back to step 1

```
Event loop tick:
  → is database response ready?  Yes → resume coroutine A → runs until await
  → is HTTP response ready?      No  → skip
  → is file read done?           Yes → resume coroutine C → runs until await
  → repeat
```

The key rule: **the event loop can only do one thing at a time**. If a coroutine runs a blocking operation (a synchronous call that doesn't yield), the entire event loop freezes — no other requests are served until it finishes.

---

## `async def` and `await`

A function defined with `async def` is a **coroutine function**. Calling it returns a coroutine object — it doesn't run until you `await` it:

```python
async def fetch_data():
    # This function can be suspended while waiting for I/O
    return "data"

# Calling it returns a coroutine — not the result yet
coro = fetch_data()

# Actually running it:
result = await fetch_data()  # "data"
```

`await` can only be used inside an `async def` function. It tells the event loop: "I'm about to do some I/O. You can run other coroutines while I wait."

```python
import asyncio
import httpx

async def get_ollama_status():
    async with httpx.AsyncClient() as client:      # async context manager
        resp = await client.get("http://localhost:11434/api/tags")  # ← suspends here
    return resp.status_code
```

While `await client.get(...)` is waiting for Ollama's response, the event loop is free to run other coroutines — other incoming requests get served.

---

## `async def` vs `def` in FastAPI

FastAPI supports both:

```python
# Sync handler — FastAPI runs this in a thread pool
@app.get("/sync")
def sync_handler():
    import time
    time.sleep(1)       # blocks, but in a thread — OK
    return {"mode": "sync"}

# Async handler — FastAPI awaits this on the event loop
@app.get("/async")
async def async_handler():
    await asyncio.sleep(1)   # suspends coroutine, event loop runs others — OK
    return {"mode": "async"}
```

**When FastAPI sees a `def` (sync) handler**, it runs it in a **thread pool executor**. This isolates the blocking call from the event loop — other requests can still be served. This is the safe fallback for synchronous code.

**When FastAPI sees an `async def` handler**, it runs it directly on the event loop. If the handler accidentally calls a blocking function, the entire server freezes for that duration.

**The rule:**

| You need to call... | Use |
|---|---|
| `await some_async_function()` | `async def` |
| A sync library (requests, sqlite3, time.sleep) | `def` (or use `run_in_threadpool`) |
| A mix — mostly sync with some async | `async def` + `asyncio.to_thread()` for blocking calls |

---

## The blocking event loop — a real example

This is the most common mistake beginners make:

```python
import requests   # SYNC library — NOT async

@app.get("/health-WRONG")
async def health_wrong():
    # BUG: requests.get() blocks the thread — the event loop freezes!
    resp = requests.get("http://localhost:11434/api/tags", timeout=5)
    return {"status": resp.status_code}
```

During `requests.get(...)`, the event loop is completely blocked. Every other request to the server queues up and waits. Under load, this would be catastrophic.

The fix — use an async HTTP client:

```python
import httpx   # ASYNC HTTP client

@app.get("/health-CORRECT")
async def health_correct():
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get("http://localhost:11434/api/tags")
    return {"status": resp.status_code}
```

Now `await client.get(...)` suspends the coroutine and releases the event loop. This is exactly what Saathi's `health_check()` does.

> **Experienced note:** This is the price of async — you need async versions of everything. Sync database drivers? Use asyncpg or SQLAlchemy async. Sync file I/O? Use aiofiles. The ecosystem has largely caught up, but it's worth checking when adopting a new library.

---

## How Saathi uses async throughout

Every handler in the Saathi API is `async def`. Here's why each one needs it:

**`health_check()`** — calls `httpx.AsyncClient` to ping Ollama. Must be async to avoid blocking.

```python
async def health_check() -> HealthResponse:
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(f"{settings.ollama_base_url}/api/tags")
```

**`chat()`** — calls `await graph.ainvoke(...)`. The LangGraph agent makes LLM calls and tool calls, all of which are async I/O. If this were sync, it would block the event loop for the entire multi-second LLM generation time.

```python
async def chat(req: ChatRequest, graph: GraphDep) -> ChatResponse:
    result = await graph.ainvoke(_build_input(req), config=...)
```

**`chat_stream()`** — uses an async generator (`async for chunk in graph.astream(...)`). Token-by-token streaming is inherently async — each token arrives as an event from Ollama.

```python
async def event_generator():
    async for chunk, _metadata in graph.astream(..., stream_mode="messages"):
        ...
        yield f"data: {payload}\n\n"
```

**`get_session_history()`** — calls `await graph.aget_state(...)` which reads from the async SQLite checkpointer.

The entire Saathi stack is async from top to bottom: FastAPI → LangGraph → langchain-ollama → httpx → Ollama.

---

## `asyncio.gather` — concurrent async tasks

Sometimes you need to run multiple async operations concurrently. `asyncio.gather` runs them all at the same time and waits for all to complete:

```python
import asyncio

async def fetch_a():
    await asyncio.sleep(1)
    return "A"

async def fetch_b():
    await asyncio.sleep(1)
    return "B"

# Sequential: takes 2 seconds
a = await fetch_a()
b = await fetch_b()

# Concurrent: takes 1 second
a, b = await asyncio.gather(fetch_a(), fetch_b())
```

This is how Saathi's code review feature works — it runs four specialist LLM reviewers concurrently with `asyncio.gather`, cutting review time to roughly 1/4 of sequential.

---

## `asyncio.to_thread` — running sync code without blocking

If you have no choice but to call a sync library from an async handler, use `asyncio.to_thread` to run it in a thread pool without blocking the event loop:

```python
import asyncio
import time

async def slow_handler():
    # time.sleep blocks — but we move it to a thread
    result = await asyncio.to_thread(time.sleep, 2)
    return {"done": True}
```

This is equivalent to FastAPI's thread pool for `def` handlers, but gives you control over which specific calls are offloaded.

---

## Summary

- The event loop handles concurrency by suspending coroutines at `await` points and running others.
- `async def` + `await` is for I/O-bound operations (HTTP calls, database, file I/O).
- Regular `def` handlers are run in a thread pool by FastAPI — safe for blocking code.
- Never call sync blocking code (requests, time.sleep, sqlite3) inside `async def` without `asyncio.to_thread`.
- The whole Saathi API is `async def` because the LangGraph agent and all its tools are async.
- `asyncio.gather` runs multiple async operations concurrently — far faster than awaiting them sequentially.

---

*Previous: [Chapter 6 — Dependency Injection](ch-06-dependency-injection.md)*  
*Next: [Chapter 8 — Routers](ch-08-routers.md)*
