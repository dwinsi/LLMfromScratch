# Chapter 9 — Streaming Responses and Server-Sent Events

> **What you'll learn:** what `StreamingResponse` is, how async generators work, the SSE wire format, how `POST /chat/stream` streams LLM tokens, and how to consume SSE from Python, curl, and JavaScript.

---

## Why streaming matters for AI APIs

When you call `POST /chat`, the server waits for the entire LLM response to be generated before sending anything back. For a short answer that's fine. For a 500-word explanation, you're staring at a blank screen for 10–30 seconds.

Streaming solves this: as the LLM generates each token, the server immediately forwards it to the client. The user sees the answer appearing word-by-word — just like ChatGPT's interface. The total time to *complete* the response is the same, but perceived latency drops dramatically.

---

## `StreamingResponse` — the FastAPI primitive

FastAPI's `StreamingResponse` takes a generator (sync or async) that `yield`s chunks of bytes or strings. It sends each chunk to the client as it's produced, without buffering the entire response first:

```python
from fastapi.responses import StreamingResponse

@app.get("/count")
def stream_numbers():
    def generate():
        for i in range(10):
            yield f"{i}\n"
    return StreamingResponse(generate(), media_type="text/plain")
```

```bash
curl http://localhost:8000/count
# 0
# 1
# 2
# ...
```

The client receives each number as it's produced. For a FastAPI async handler you'd use an `async def` generator:

```python
import asyncio

@app.get("/count-async")
async def stream_numbers_async():
    async def generate():
        for i in range(10):
            await asyncio.sleep(0.1)   # simulate work
            yield f"{i}\n"
    return StreamingResponse(generate(), media_type="text/plain")
```

---

## Server-Sent Events — the protocol

**SSE (Server-Sent Events)** is a simple protocol built on top of plain HTTP. The server keeps the connection open and sends events as text. Each event follows this format:

```
data: <payload>\n\n
```

The double newline `\n\n` terminates each event. That's it — no binary encoding, no special handshake. Browsers have a built-in `EventSource` API for consuming SSE. In Python, any HTTP client that can stream responses can read them.

Optional SSE fields (rarely needed):

```
event: custom-event-name\n
id: 42\n
data: {"message": "hello"}\n\n
```

Saathi uses only the `data:` field, with JSON payloads:

```
data: {"session_id": "abc...", "delta": "The ", "done": false}\n\n
data: {"session_id": "abc...", "delta": "agent ", "done": false}\n\n
data: {"session_id": "abc...", "delta": "uses ", "done": false}\n\n
data: {"session_id": "abc...", "delta": "", "done": true}\n\n
```

Each `delta` is a token (or small group of tokens) from the LLM. The final event has `"done": true` to signal end of stream.

---

## `POST /chat/stream` — complete walkthrough

```python
# routes/chat.py

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, graph: GraphDep):
    """
    Stream the agent's reply token-by-token using Server-Sent Events.
    """

    async def event_generator():
        async for chunk, _metadata in graph.astream(
            _build_input(req),
            config=_thread_config(req.session_id),
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                delta = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                payload = json.dumps(
                    {"session_id": req.session_id, "delta": delta, "done": False}
                )
                yield f"data: {payload}\n\n"

        # Terminal event — signals the client that streaming is complete
        yield f"data: {json.dumps({'session_id': req.session_id, 'delta': '', 'done': True})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
```

**Breaking it down:**

**`async def event_generator()`** — an async generator function. It `yield`s strings (SSE events). It's defined inside `chat_stream` so it can close over `req` and `graph` from the outer scope.

**`graph.astream(..., stream_mode="messages")`** — LangGraph's streaming mode. With `stream_mode="messages"`, it yields `(message_chunk, metadata)` tuples as the LLM produces tokens. This is fundamentally different from `ainvoke` which only yields the final complete state.

**`isinstance(chunk, AIMessageChunk)`** — during the ReAct loop, LangGraph also streams tool call messages and tool results. We only care about the AI's final text response — `AIMessageChunk` is what comes from the LLM token stream. Tool messages are filtered out.

**`f"data: {payload}\n\n"`** — SSE format: `data: ` prefix, JSON payload, double newline.

**Terminal event** — after the `async for` loop finishes (stream is exhausted), we send one final event with `"done": true`. Clients should stop reading when they see this. Without it, some clients keep the connection open waiting for more events.

**`media_type="text/event-stream"`** — tells the client this is an SSE stream. Browsers use this to activate the `EventSource` API. If you set `media_type="text/plain"`, browsers won't automatically reconnect on disconnect.

**`"Cache-Control": "no-cache"`** — prevents proxies and browsers from buffering the stream. Without this, some proxies hold the entire response before forwarding.

**`"X-Accel-Buffering": "no"`** — tells nginx (if proxied) not to buffer this response. Essential in production.

---

## Why there's no `response_model`

Notice the streaming endpoint has no `response_model=`:

```python
@router.post("/chat/stream")       # ← no response_model
async def chat_stream(req: ChatRequest, graph: GraphDep):
    return StreamingResponse(...)
```

`response_model` works by taking the handler's return value, passing it through Pydantic, and serialising to JSON. But `StreamingResponse` is already the response — FastAPI doesn't touch it. The schema documentation for streaming endpoints is usually written as a docstring or in the `response_description` parameter.

---

## Consuming SSE — curl

```bash
# -N disables buffering so you see output as it arrives
curl -N -X POST http://localhost:8000/chat/stream \
     -H "Content-Type: application/json" \
     -d '{"message": "Explain the LangGraph state graph in this project"}'
```

You'll see events printing to the terminal as they arrive:

```
data: {"session_id": "...", "delta": "The", "done": false}
data: {"session_id": "...", "delta": " LangGraph", "done": false}
...
data: {"session_id": "...", "delta": "", "done": true}
```

---

## Consuming SSE — Python

```python
import httpx
import json

def stream_chat(message: str):
    with httpx.Client(timeout=None) as client:
        with client.stream(
            "POST",
            "http://localhost:8000/chat/stream",
            json={"message": message},
        ) as resp:
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = json.loads(line[6:])   # strip "data: "
                print(chunk["delta"], end="", flush=True)
                if chunk["done"]:
                    break
    print()  # newline after stream ends

stream_chat("What Python files are in this project?")
```

`httpx.Client` with `stream()` keeps the connection open and iterates lines as they arrive. `timeout=None` prevents the connection from timing out during a long response.

---

## Consuming SSE — JavaScript (browser)

```javascript
const response = await fetch('http://localhost:8000/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: 'Explain this codebase' }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = '';

while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop();  // keep incomplete last chunk

    for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const chunk = JSON.parse(line.slice(6));
        document.getElementById('output').textContent += chunk.delta;
        if (chunk.done) return;
    }
}
```

Alternatively, for simple cases without POST bodies, the browser's `EventSource` API works:

```javascript
const es = new EventSource('http://localhost:8000/events');
es.onmessage = (event) => {
    const chunk = JSON.parse(event.data);
    console.log(chunk.delta);
    if (chunk.done) es.close();
};
```

`EventSource` only supports GET requests — use the `fetch` + `ReadableStream` approach for POST endpoints like Saathi's.

---

## Summary

- `StreamingResponse` takes an (async) generator and streams chunks to the client as they're produced.
- SSE format is `data: <payload>\n\n` — simple, human-readable, supported natively in browsers.
- Use `media_type="text/event-stream"` and `"Cache-Control": "no-cache"` headers.
- `"X-Accel-Buffering": "no"` prevents nginx from buffering your stream in production.
- There is no `response_model` on streaming endpoints — the `StreamingResponse` is returned directly.
- Consume SSE with `curl -N`, `httpx.Client.stream()` in Python, or `fetch` + `ReadableStream` in JS.

---

*Previous: [Chapter 8 — Routers](ch-08-routers.md)*  
*Next: [Chapter 10 — Lifespan](ch-10-lifespan.md)*
