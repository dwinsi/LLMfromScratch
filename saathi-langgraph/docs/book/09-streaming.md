# Chapter 9 — Streaming: Real-Time Token Output

> **About this chapter**
> This chapter covers how saathi-langgraph delivers token-by-token output to the
> terminal as the LLM generates its response. We start from the user-experience
> rationale, work down through the LangGraph streaming API, and finish with
> practical patterns for buffering, interruption, testing, and Rich console
> integration. Code examples are drawn directly from the saathi source tree so
> you can follow along with a real implementation.

---

## Table of Contents

1. Why Streaming Matters for UX
2. LLM Streaming Mechanics
3. `astream_events` — the LangGraph Streaming API
4. Token Chunk Assembly
5. Buffering for Tool Calls
6. `version="v2"` vs `version="v1"`
7. Interleaving Tools and Text
8. Token Usage from Streaming
9. Rich Console Integration
10. `--print` Mode and Streaming
11. Interrupting Streams
12. Streaming in Tests
13. `astream` (State Streaming)
14. Summary
15. Key Takeaways

---

## 9.1  Why Streaming Matters for UX

### 9.1.1  The Blank-Screen Problem

Imagine asking a colleague a question and they say nothing for twenty seconds,
then suddenly recite a three-paragraph answer in one breath. That is what a
non-streaming LLM feels like to the user. The screen sits blank. The cursor
blinks. Nothing happens for ten, fifteen, sometimes thirty seconds. Then the
full response materialises at once.

Most users interpret this silence as the application being broken. They hit
Enter again. They switch windows. They lose confidence in the tool. The
first-time experience — the all-important impression that determines whether
someone keeps using a CLI tool — is poisoned by a wall of waiting followed by
a wall of text.

Streaming fixes this. Tokens appear on the screen as the LLM generates them.
The user sees progress immediately. They can start reading while the model is
still writing. If the response heads in the wrong direction they can press
Ctrl+C and redirect the conversation without waiting for completion. The
application feels alive.

### 9.1.2  First-Token Latency vs. Time-to-Complete

There are two distinct latency numbers that matter for LLM UX:

**First-token latency (FTL)** — the elapsed wall-clock time from sending the
request until the first token appears on screen. This is dominated by:

- Network round-trip to the model server (negligible for local Ollama)
- Model loading time (if the model is not already resident in VRAM)
- The prefill pass: the model encodes the entire prompt before it can start
  generating. Long prompts mean longer prefill.

**Time-to-complete (TTC)** — the elapsed wall-clock time from the first token
until the last token. This is determined almost entirely by:

- Model size (number of parameters)
- Hardware (GPU vs CPU, VRAM bandwidth)
- Response length (more tokens = more time)

Without streaming, the user perceives only TTC, and TTC includes FTL. The
subjective experience is dominated by the worst number. With streaming, the
user's perceived latency is only FTL — the time until *something* appears —
and FTL for a local Ollama model is often under one second.

```text
Without streaming:

  t=0s   ----[ silence ]-----[ silence ]-----[ silence ]----  t=18s
                                                               ^
                                                         full text appears

With streaming:

  t=0s   [first token at ~0.4s] ......... [last token at 18s]
          ^                                ^
      user starts reading             completion
```

The total compute time is identical. The *experience* is completely different.

> **Takeaway:** Streaming does not make the model faster. It makes the
> application feel responsive by surfacing the latency that already exists
> inside the generation process.

### 9.1.3  Streaming as a Design Constraint

Choosing to stream shapes the rest of your architecture. It means:

- Your graph invocation must be `async` throughout.
- You cannot simply `await graph.ainvoke(...)` and print the result; you must
  iterate over an async generator.
- Buffering is your responsibility: if you want to display partial tool-call
  arguments, you must accumulate chunks yourself.
- Error handling must work mid-stream, not just at the end.

saathi-langgraph was designed with streaming as a first-class requirement from
the start. The `cli.py` entry point, the `display.py` helpers, and the graph
configuration all assume streaming. The non-streaming `ainvoke` path exists
only for unit tests.

---

## 9.2  LLM Streaming Mechanics

### 9.2.1  How Token Generation Works

A large language model generates text one token at a time. A token is
typically two to four characters of English text (shorter for common words,
longer for rare or technical terms). The model cannot compute token N+1 until
token N is finalised, because token N becomes part of the key-value cache that
speeds up subsequent generation.

This sequential dependency is both the reason streaming is possible and the
reason it cannot be parallelised. The model genuinely produces tokens one at a
time, so each token can be sent to the client as soon as it is produced.

```text
Prompt tokens (prefill):
  [the][user][asked][about][streaming]  →  prefill pass (parallel)

Generation tokens (autoregressive):
  [Streaming]  →  [ is]  →  [ a]  →  [ technique]  →  [ that] ...
     t=0.1s      t=0.2s    t=0.3s      t=0.4s          t=0.5s

Each token is emitted to the client as soon as it is sampled.
```

### 9.2.2  HTTP Transport for Streaming

The HTTP 1.1 protocol supports streaming via *chunked transfer encoding*. The
server does not send a `Content-Length` header; instead it sends the response
body in variable-size chunks, each preceded by its size in hexadecimal. The
connection stays open until the server sends a zero-length chunk.

Modern LLM APIs typically layer a higher-level protocol on top of this. The
most common choice is *Server-Sent Events* (SSE), which is an HTML5 standard
for server-push streams:

```text
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache

data: {"token": "Stream"}

data: {"token": "ing"}

data: {"token": " is"}

data: [DONE]
```

Each event begins with `data:` and ends with a blank line. The `[DONE]`
sentinel signals end of stream.

### 9.2.3  How Ollama Streams

Ollama uses a simpler format: newline-delimited JSON (NDJSON). Each line of
the response body is a complete JSON object. The `done` field is `false` for
every intermediate chunk and `true` for the final chunk.

```text
{"model":"llama3","created_at":"...","message":{"role":"assistant","content":"Stream"},"done":false}
{"model":"llama3","created_at":"...","message":{"role":"assistant","content":"ing"},"done":false}
{"model":"llama3","created_at":"...","message":{"role":"assistant","content":" is"},"done":false}
{"model":"llama3","created_at":"...","message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":22,"eval_count":47}
```

Notice that the final chunk (where `done` is `true`) carries additional
metadata: `prompt_eval_count` (tokens in the prompt) and `eval_count` (tokens
generated). This is where Ollama surfaces token usage, and we will return to
it in section 9.8.

The LangChain `ChatOllama` integration handles the HTTP connection and JSON
parsing internally. By the time events reach your LangGraph streaming loop,
they have already been transformed into LangChain `AIMessageChunk` objects.
You never see raw NDJSON in your application code.

### 9.2.4  The LangChain Abstraction Layer

LangChain provides a unified streaming interface across many LLM providers.
The `BaseChatModel.astream()` method is an async generator that yields
`AIMessageChunk` objects regardless of whether the underlying provider uses
SSE, NDJSON, WebSockets, or any other transport.

```flow
                         ┌──────────────────┐
                         │   Your Code      │
                         │  async for chunk │
                         └────────┬─────────┘
                                  │ AIMessageChunk
                         ┌────────▼─────────┐
                         │  LangChain       │
                         │  ChatOllama      │
                         └────────┬─────────┘
                                  │ NDJSON
                         ┌────────▼─────────┐
                         │  Ollama Server   │
                         │  localhost:11434 │
                         └──────────────────┘
```

LangGraph adds another layer on top: it wraps the model call inside a graph
node, adds routing and state management, and exposes a streaming API that
emits *events* rather than raw chunks. Events carry contextual metadata (which
node produced this chunk? which tool is running?) that raw chunks do not.

---

## 9.3  `astream_events` — the LangGraph Streaming API

### 9.3.1  Overview

`graph.astream_events()` is the primary streaming API for LangGraph
applications. It returns an async generator that yields dictionaries. Each
dictionary represents a *lifecycle event* in the graph execution: a node
starting, a model generating a token, a tool returning a result.

The signature is:

```python
async def astream_events(
    input: Any,
    config: Optional[RunnableConfig] = None,
    *,
    version: str = "v2",
    include_names: Optional[Sequence[str]] = None,
    include_types: Optional[Sequence[str]] = None,
    include_tags: Optional[Sequence[str]] = None,
    exclude_names: Optional[Sequence[str]] = None,
    exclude_types: Optional[Sequence[str]] = None,
    exclude_tags: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> AsyncIterator[StreamEvent]:
```

The filtering parameters (`include_names`, `exclude_types`, etc.) let you
narrow the event stream to only what you need. For a typical CLI that just
wants to print tokens, you only care about `on_chat_model_stream` events, but
observing all events during development gives you invaluable insight into graph
execution.

### 9.3.2  Event Structure

Every event is a dictionary with the following fields:

| Field      | Type   | Description                                              |
| ------------ | -------- | ---------------------------------------------------------- |
| `event`    | `str`  | The event type name (see below)                          |
| `name`     | `str`  | The name of the runnable that produced this event        |
| `run_id`   | `str`  | UUID identifying this particular run of the runnable     |
| `parent_run_id` | `str` or `None` | The run_id of the parent runnable, if any   |
| `tags`     | `list[str]` | Tags attached to the runnable                       |
| `metadata` | `dict` | Metadata from the runnable config                        |
| `data`     | `dict` | Event-specific payload (varies by event type)            |

Here is a concrete example of an `on_chat_model_stream` event printed as a
Python dictionary:

```python
{
    "event": "on_chat_model_stream",
    "name": "ChatOllama",
    "run_id": "a3f2c1d0-4b5e-6f7a-8b9c-0d1e2f3a4b5c",
    "parent_run_id": "b4e3d2c1-5c6d-7e8f-9a0b-1c2d3e4f5a6b",
    "tags": ["seq:step:1"],
    "metadata": {
        "langgraph_step": 2,
        "langgraph_node": "agent",
        "langgraph_triggers": ["messages"],
        "langgraph_task_idx": 0,
        "checkpoint_id": "...",
        "checkpoint_ns": "",
        "ls_model_name": "llama3",
        "ls_model_type": "chat",
        "ls_temperature": 0.0,
    },
    "data": {
        "chunk": AIMessageChunk(
            content="Stream",
            id="run-a3f2c1d0-...",
        )
    }
}
```

The `metadata` dictionary is particularly rich. `langgraph_node` tells you
which node in the graph produced this event. `ls_model_name` tells you which
Ollama model is running. This metadata is only available in `version="v2"`.

### 9.3.3  Event Types Reference

LangGraph/LangChain define the following event types:

**Model events:**

- `on_chat_model_start` — the LLM received its input messages and started
  generating. `data["input"]` contains the messages.
- `on_chat_model_stream` — a token chunk was generated. `data["chunk"]`
  is an `AIMessageChunk`.
- `on_chat_model_end` — the LLM finished generating. `data["output"]` is
  the complete `AIMessage`.

**Tool events:**

- `on_tool_start` — a tool is about to execute. `data["input"]` contains
  the tool's input arguments.
- `on_tool_end` — a tool finished executing. `data["output"]` contains the
  tool's return value.
- `on_tool_error` — a tool raised an exception. `data["error"]` contains
  the exception.

**Chain/node events:**

- `on_chain_start` — a chain or graph node started. `data["input"]` is the
  node's input state.
- `on_chain_stream` — a chain emitted an intermediate result.
- `on_chain_end` — a chain or node finished. `data["output"]` is the
  node's output.

**Retriever events** (for RAG applications):

- `on_retriever_start`
- `on_retriever_end`
- `on_retriever_error`

saathi primarily handles `on_chat_model_stream`, `on_tool_start`, and
`on_tool_end`. The chain events are useful for debugging but are not surfaced
to the user in normal operation.

### 9.3.4  A Minimal Streaming Loop

The simplest possible streaming loop that prints tokens:

```python
import asyncio
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama

async def stream_response(graph, state, config):
    async for event in graph.astream_events(state, config, version="v2"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                print(chunk.content, end="", flush=True)
    print()  # newline after the response

asyncio.run(stream_response(graph, initial_state, config))
```

`flush=True` is important. Without it, Python's buffered output may hold
several chunks before writing to the terminal, defeating the purpose of
streaming. `end=""` prevents `print` from adding a newline after each token.

---

## 9.4  Token Chunk Assembly

### 9.4.1  The `AIMessageChunk` Object

Each `on_chat_model_stream` event carries an `AIMessageChunk` in
`event["data"]["chunk"]`. `AIMessageChunk` is a subclass of `AIMessage` that
is designed to be accumulated.

The key fields on `AIMessageChunk`:

| Field               | Type                     | Description                           |
| --------------------- | -------------------------- | --------------------------------------- |
| `content`           | `str` or `list`          | The text content of this chunk        |
| `tool_call_chunks`  | `list[ToolCallChunk]`    | Partial tool call data (if any)       |
| `id`                | `str`                    | Stable run ID for this generation     |
| `usage_metadata`    | `dict` or `None`         | Token counts (only on final chunk)    |
| `response_metadata` | `dict`                   | Raw provider response metadata        |

The `content` field is almost always a plain string for text generation. It
can be a list of content blocks for providers that use structured content
(e.g., Anthropic's Claude with image inputs), but for Ollama it is always a
string.

### 9.4.2  Accumulating Chunks

LangChain `AIMessageChunk` objects support the `+` operator for accumulation.
You can build the complete message by summing all chunks:

```python
from langchain_core.messages import AIMessageChunk

accumulated: AIMessageChunk | None = None

async for event in graph.astream_events(state, config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        chunk: AIMessageChunk = event["data"]["chunk"]
        if accumulated is None:
            accumulated = chunk
        else:
            accumulated = accumulated + chunk

# accumulated is now a complete AIMessage (as AIMessageChunk)
full_text = accumulated.content
```

This is useful when you need the complete response for post-processing (e.g.,
syntax highlighting a code block after streaming, or extracting structured
data). For simple printing, you do not need to accumulate — just emit each
chunk's content as it arrives.

### 9.4.3  The Streaming Loop in saathi's CLI

Here is the core streaming loop from saathi's `cli.py`, showing how tokens
are printed to the terminal:

```python
async for event in graph.astream_events(state, config, version="v2"):
    kind = event["event"]

    if kind == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        # chunk.content is a string for Ollama; empty string for
        # tool-call-only chunks
        if isinstance(chunk.content, str) and chunk.content:
            print(chunk.content, end="", flush=True)

    elif kind == "on_tool_start":
        # Print a newline to separate the text from the tool indicator
        print()
        tool_name = event["name"]
        print(f"[tool: {tool_name}]", flush=True)

    elif kind == "on_tool_end":
        # Optionally show tool result summary
        pass

# Final newline after the complete response
print()
```

Notice that we check `isinstance(chunk.content, str)` before printing. When
the model is generating a tool call instead of text, the `content` field of
each chunk may be an empty string, while the actual tool call data lives in
`chunk.tool_call_chunks`. We discuss this in section 9.5.

### 9.4.4  Why `flush=True` Matters

Python's standard output is line-buffered when connected to a terminal (it
flushes automatically on newlines) but block-buffered when connected to a
pipe. Since `print(chunk.content, end="")` never emits a newline, the buffer
would fill up and the output would appear in bursts rather than token by token.

`flush=True` forces a `sys.stdout.flush()` call after each print. This
ensures that every token is delivered to the terminal immediately, at the cost
of one additional system call per token. For a typical generation rate of 30-80
tokens per second, this overhead is negligible.

If you are using the `rich` library's `Console` object (section 9.9), the
equivalent is to pass `no_color=False` and ensure the console's internal
buffer is configured for immediate flush:

```python
from rich.console import Console

console = Console(highlight=False)

# Rich flushes after each print by default when not capturing
console.print(chunk.content, end="")
```

### 9.4.5  Handling Multi-line Chunks

Ollama typically emits tokens of one to four characters. Occasionally a chunk
may contain a newline character embedded in it, for example when the model
produces a markdown code fence. The streaming loop handles this naturally
because we are printing raw characters without any line-aware processing.

If you need line-aware processing (for example, to apply syntax highlighting
to complete lines), you would need to buffer until you see a newline:

```python
line_buffer = ""

async for event in graph.astream_events(state, config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        if isinstance(chunk.content, str):
            line_buffer += chunk.content
            while "\n" in line_buffer:
                line, line_buffer = line_buffer.split("\n", 1)
                process_complete_line(line)

# Don't forget the final partial line
if line_buffer:
    process_complete_line(line_buffer)
```

saathi does not do line-aware processing during streaming; syntax highlighting
is applied retrospectively to code blocks in the history display. This keeps
the streaming path simple and fast.

---

## 9.5  Buffering for Tool Calls

### 9.5.1  Tool Calls Are Streamed Too

When the LLM decides to call a tool, it does not emit a text token. Instead,
it emits a structured tool call that specifies the tool name and its JSON
arguments. In a streaming context, this tool call arrives in pieces across
multiple chunks, just like text content.

The `AIMessageChunk` object has a `tool_call_chunks` field for exactly this
purpose. Each element of `tool_call_chunks` is a `ToolCallChunk`:

```python
@dataclass
class ToolCallChunk:
    name: str | None       # None for subsequent chunks of the same call
    args: str              # A fragment of the JSON arguments string
    id: str | None         # None for subsequent chunks
    index: int | None      # Position in the tool_calls list
    type: str              # Always "tool_call_chunk"
```

Here is an example sequence of chunks when the model calls a hypothetical
`search` tool with `{"query": "LangGraph streaming"}`:

```text
Chunk 1: tool_call_chunks=[ToolCallChunk(name="search", args='', id="call_abc", index=0)]
Chunk 2: tool_call_chunks=[ToolCallChunk(name=None, args='{"qu', id=None, index=0)]
Chunk 3: tool_call_chunks=[ToolCallChunk(name=None, args='ery":', id=None, index=0)]
Chunk 4: tool_call_chunks=[ToolCallChunk(name=None, args=' "Lang', id=None, index=0)]
Chunk 5: tool_call_chunks=[ToolCallChunk(name=None, args='Graph streaming"}', id=None, index=0)]
```

The first chunk establishes the call ID and tool name. Subsequent chunks
carry fragments of the JSON argument string. You accumulate the `args` strings
to recover the complete argument JSON.

### 9.5.2  Detecting "Text Mode" vs "Tool Call Mode"

During streaming, the model is in one of two modes at any given moment:

1. **Text generation mode**: `chunk.content` is a non-empty string,
   `chunk.tool_call_chunks` is empty.
2. **Tool call mode**: `chunk.content` is an empty string (or absent),
   `chunk.tool_call_chunks` is non-empty.

A model *can* transition from text to tool call mid-generation (emit some
reasoning text, then decide to call a tool), though in practice most models
make this decision at the start of a generation step.

Here is a detection pattern:

```python
is_generating_text = False
is_generating_tool_call = False

async for event in graph.astream_events(state, config, version="v2"):
    if event["event"] != "on_chat_model_stream":
        continue

    chunk = event["data"]["chunk"]

    # Detect mode
    has_text = isinstance(chunk.content, str) and len(chunk.content) > 0
    has_tool_call = len(chunk.tool_call_chunks) > 0

    if has_text and not is_generating_text:
        is_generating_text = True
        is_generating_tool_call = False
        # Optional: print a visual indicator that text output is starting

    if has_tool_call and not is_generating_tool_call:
        is_generating_tool_call = True
        is_generating_text = False
        # Print the tool name when we first see it
        tool_name = chunk.tool_call_chunks[0].name
        if tool_name:
            print(f"\n[Calling tool: {tool_name}...]", flush=True)

    if has_text:
        print(chunk.content, end="", flush=True)
```

### 9.5.3  Accumulating Tool Call Chunks

If you need to inspect the complete tool call arguments before the tool
executes (for example, to display them to the user), you must accumulate the
`args` fragments yourself:

```python
from collections import defaultdict

# Maps tool call ID -> accumulated args string
pending_tool_calls: dict[str, dict] = defaultdict(lambda: {"name": None, "args": ""})

async for event in graph.astream_events(state, config, version="v2"):
    if event["event"] != "on_chat_model_stream":
        continue

    chunk = event["data"]["chunk"]

    for tc_chunk in chunk.tool_call_chunks:
        call_id = tc_chunk.id or "unknown"
        if tc_chunk.name:
            pending_tool_calls[call_id]["name"] = tc_chunk.name
        pending_tool_calls[call_id]["args"] += tc_chunk.args or ""

# After streaming, pending_tool_calls contains complete tool calls
for call_id, call_data in pending_tool_calls.items():
    import json
    args = json.loads(call_data["args"])
    print(f"Tool: {call_data['name']}, Args: {args}")
```

In saathi, this level of tool call introspection is not needed in the main
streaming loop because LangGraph handles tool execution automatically.
The `on_tool_start` event fires when the tool actually runs, and by that
point LangGraph has already assembled the complete arguments. saathi simply
listens to `on_tool_start` and displays the tool name to the user.

### 9.5.4  Parallel Tool Calls

Some models can decide to call multiple tools simultaneously (parallel tool
calls). In the streaming response, this appears as multiple `tool_call_chunks`
entries with different `index` values in the same chunk, or in subsequent
chunks.

The detection pattern generalises:

```python
pending_tool_calls: dict[int, dict] = defaultdict(lambda: {"name": None, "args": ""})

for tc_chunk in chunk.tool_call_chunks:
    idx = tc_chunk.index or 0
    if tc_chunk.name:
        pending_tool_calls[idx]["name"] = tc_chunk.name
    pending_tool_calls[idx]["args"] += tc_chunk.args or ""
```

Using `index` as the key rather than `id` handles the case where some
providers do not emit a stable `id` for each chunk.

---

## 9.6  `version="v2"` vs `version="v1"`

### 9.6.1  History

LangGraph's `astream_events` was introduced with API version `"v1"`. As the
library matured, the event format was revised to provide richer metadata and
fix some edge cases. The revised format is `"v2"`. Both versions coexist in
the current LangGraph codebase for backward compatibility.

saathi uses `version="v2"` exclusively. If you are maintaining an older
LangGraph application, you may still be using v1.

### 9.6.2  What Changed in v2

**LangGraph node metadata in `metadata`:**

In v1, the `metadata` dictionary may not include `langgraph_node`,
`langgraph_step`, or `langgraph_triggers`. In v2, these fields are always
present when the event originates from inside a LangGraph graph.

```python
# v1 event metadata (partial)
{
    "ls_model_name": "llama3",
    "ls_model_type": "chat",
}

# v2 event metadata (partial)
{
    "ls_model_name": "llama3",
    "ls_model_type": "chat",
    "langgraph_step": 2,
    "langgraph_node": "agent",
    "langgraph_triggers": ["messages"],
    "langgraph_task_idx": 0,
    "checkpoint_id": "1ef8...",
    "checkpoint_ns": "",
}
```

This metadata is invaluable for multi-agent graphs where the same model may
be called from different nodes. The `langgraph_node` field tells you exactly
which node in the graph produced the event.

**Consistent event naming:**

In v1, some event names were inconsistent between different runnable types.
v2 standardises the naming across all runnable types.

**Subgraph events:**

In v2, events from nested subgraphs are properly namespaced. If your graph
calls a subgraph, events from the subgraph include a `checkpoint_ns` that
identifies the subgraph context. In v1, subgraph events could be
indistinguishable from top-level graph events.

### 9.6.3  Migrating from v1 to v2

If you have existing code using v1, the primary change is:

1. Pass `version="v2"` to `astream_events`.
2. Update any code that reads `event["metadata"]` to use the new field names.
3. Update any code that relies on the old `run_id` nesting behaviour for
   subgraphs.

The actual event type names (`on_chat_model_stream`, `on_tool_start`, etc.)
are unchanged between v1 and v2.

### 9.6.4  Why saathi Always Uses v2

saathi targets LangGraph 0.2+ which ships with v2 as the recommended API.
The richer metadata enables features like:

- Displaying which agent node is currently running in a multi-agent setup.
- Filtering events to only those from specific nodes (e.g., only stream
  tokens from the "responder" node, not the "planner" node).
- Debugging graph execution by correlating events to specific graph steps.

If you are building a new LangGraph application, use `version="v2"` from
the start.

---

## 9.7  Interleaving Tools and Text

### 9.7.1  The Pattern

A ReAct-style agent (the pattern saathi uses) generates responses in a loop:

1. The model generates some text (reasoning or partial answer).
2. The model calls a tool to get information it needs.
3. The tool result is added to the conversation.
4. The model generates more text, possibly calling more tools.
5. Eventually the model generates a final answer with no tool calls.

In a streaming context, this means the event stream for a single user turn
looks like:

```text
on_chat_model_start       (model starts generating)
on_chat_model_stream      (text: "Let me look that up.")
on_chat_model_stream      (tool call: search)
on_chat_model_end         (generation 1 complete)
on_tool_start             (search tool runs)
on_tool_end               (search tool returns results)
on_chat_model_start       (model starts second generation)
on_chat_model_stream      (text: "Based on the search results, ...")
on_chat_model_stream      (text: "the answer is ...")
on_chain_end              (graph complete)
```

The streaming loop must handle this interleaving gracefully.

### 9.7.2  Tracking State in the Streaming Loop

```python
class StreamingState:
    """Tracks the current mode of the streaming loop."""
    def __init__(self):
        self.in_tool_call = False
        self.last_was_text = False
        self.tool_call_count = 0

state = StreamingState()

async for event in graph.astream_events(graph_input, config, version="v2"):
    kind = event["event"]

    if kind == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        content = chunk.content

        if isinstance(content, str) and content:
            if state.in_tool_call:
                # Transitioning from tool call back to text output
                # Print a newline to visually separate the tool result
                print()
                state.in_tool_call = False

            print(content, end="", flush=True)
            state.last_was_text = True

        elif chunk.tool_call_chunks:
            # Model is generating a tool call
            state.in_tool_call = True
            # Optionally display tool name on first chunk
            for tc in chunk.tool_call_chunks:
                if tc.name:
                    print(f"\n  [calling: {tc.name}]", flush=True)

    elif kind == "on_tool_start":
        tool_name = event["name"]
        if not state.in_tool_call:
            print()
        print(f"  [tool {tool_name}: running...]", flush=True)
        state.tool_call_count += 1
        state.in_tool_call = True

    elif kind == "on_tool_end":
        print(f"  [tool {event['name']}: done]", flush=True)

    elif kind == "on_chain_end" and event["name"] == "LangGraph":
        # Top-level graph finished
        if state.last_was_text:
            print()  # Final newline
```

### 9.7.3  Visual Design for Tool Calls

The design of saathi's tool call indicators follows a few principles:

**Indent tool calls visually.** The main text output is flush left. Tool call
indicators are indented with two spaces. This creates a visual hierarchy that
makes it easy to distinguish model reasoning from tool execution.

**Bracket tool names.** Using square brackets like `[tool: search]` is
conventional in CLI tools for status indicators. It signals "this is a system
message, not model output."

**Use a different colour when available.** In Rich console mode (section 9.9),
saathi renders tool call indicators in a muted colour (e.g., `dim cyan`) so
they do not compete visually with the model's text output.

**Do not block on tool results.** The streaming loop does not wait for the
tool result to be "interesting" before continuing to display output. If the
tool is fast (e.g., a local Python function), the pause is imperceptible. If
the tool is slow (e.g., a web search), the user sees the indicator and knows
why there is a pause.

### 9.7.4  Example: A Complete Interleaved Interaction

```text
User: What is the capital of France, and what time is it there now?

saathi:
  [calling: get_current_time]
  [tool get_current_time: running...]
  [tool get_current_time: done]

The capital of France is Paris. The current time in Paris is
14:32 CEST (Central European Summer Time), which is UTC+2.
```

The first generation step produced only a tool call (the model did not emit
any text before deciding to call the tool). The second generation step
produced the complete answer. The streaming loop handled the transition
transparently.

---

## 9.8  Token Usage from Streaming

### 9.8.1  Why Token Counts Matter

Token counts are important for:

- **Cost tracking**: Even for local models, knowing how many tokens are
  processed helps you understand performance characteristics.
- **Context window management**: If you are getting close to the model's
  context limit, you need to know the prompt token count.
- **Debugging**: Unexpectedly high token counts can signal prompt injection,
  runaway tool calls, or misconfigured system prompts.

### 9.8.2  Where Token Counts Appear in Streaming

For Ollama, token counts are available in the final NDJSON chunk (where
`done` is `true`). LangChain surfaces these counts in two places:

1. **`on_chat_model_end` event**: `event["data"]["output"].usage_metadata`
   contains a dict with `input_tokens`, `output_tokens`, and `total_tokens`.

2. **`AIMessageChunk.usage_metadata`**: The final chunk in the stream may
   carry `usage_metadata`. However, "final chunk" detection is not
   straightforward in a streaming context.

The most reliable approach is to listen to `on_chat_model_end`:

```python
async for event in graph.astream_events(state, config, version="v2"):
    if event["event"] == "on_chat_model_end":
        output = event["data"]["output"]
        if hasattr(output, "usage_metadata") and output.usage_metadata:
            usage = output.usage_metadata
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)
            print(
                f"\n[tokens: {prompt_tokens} in, {completion_tokens} out]",
                file=sys.stderr,
            )
```

### 9.8.3  Ollama-Specific Token Fields

Ollama's `response_metadata` on the final `AIMessage` contains the raw fields
from Ollama's API:

```python
output.response_metadata == {
    "model": "llama3",
    "created_at": "2024-01-15T10:23:45.123Z",
    "message": {"role": "assistant", "content": ""},
    "done_reason": "stop",
    "done": True,
    "total_duration": 2847394000,    # nanoseconds
    "load_duration": 1234567,        # nanoseconds
    "prompt_eval_count": 142,        # input tokens
    "prompt_eval_duration": 198000000,  # nanoseconds
    "eval_count": 89,                # output tokens
    "eval_duration": 2600000000,     # nanoseconds
}
```

The `eval_count` and `prompt_eval_count` fields are directly from Ollama.
For cases where `usage_metadata` is not available, you can fall back to these
Ollama-specific fields.

### 9.8.4  saathi's `extract_usage()` Function

saathi centralises token extraction in `usage.py`. The function handles both
the standard LangChain `usage_metadata` field and the Ollama-specific
`response_metadata` fields:

```python
# src/saathi/usage.py

from __future__ import annotations
from typing import Any
from langchain_core.messages import AIMessage


def extract_usage(message: AIMessage) -> dict[str, int]:
    """
    Extract token usage from an AIMessage, trying multiple sources.

    Returns a dict with keys: input_tokens, output_tokens, total_tokens.
    All values default to 0 if not available.
    """
    usage: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    # Primary: LangChain usage_metadata (standardised across providers)
    if message.usage_metadata:
        usage["input_tokens"] = message.usage_metadata.get("input_tokens", 0)
        usage["output_tokens"] = message.usage_metadata.get("output_tokens", 0)
        usage["total_tokens"] = message.usage_metadata.get("total_tokens", 0)
        return usage

    # Fallback: Ollama response_metadata
    resp_meta = getattr(message, "response_metadata", {}) or {}
    if "prompt_eval_count" in resp_meta:
        usage["input_tokens"] = resp_meta.get("prompt_eval_count", 0)
        usage["output_tokens"] = resp_meta.get("eval_count", 0)
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

    return usage


def format_usage(usage: dict[str, int]) -> str:
    """Format token usage for display."""
    if usage["total_tokens"] == 0:
        return ""
    return (
        f"[{usage['input_tokens']}→{usage['output_tokens']} tokens, "
        f"{usage['total_tokens']} total]"
    )
```

### 9.8.5  Integrating Usage Tracking into the Streaming Loop

```python
last_usage: dict[str, int] = {}

async for event in graph.astream_events(state, config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        if isinstance(chunk.content, str) and chunk.content:
            print(chunk.content, end="", flush=True)

    elif event["event"] == "on_chat_model_end":
        output = event["data"]["output"]
        last_usage = extract_usage(output)

print()  # final newline

# Display usage in a subtle way after the response
if last_usage and last_usage["total_tokens"] > 0:
    print(format_usage(last_usage), file=sys.stderr)
```

Writing usage to `stderr` (not `stdout`) ensures it does not pollute the
response when saathi is used in pipeline mode (`saathi --print | grep ...`).

### 9.8.6  Cumulative Usage Across Multiple Tool Calls

In a multi-step ReAct loop, the model may be called multiple times in a
single user turn. Each call generates its own `on_chat_model_end` event with
its own usage counts. To get the total usage for the entire user turn, you
must accumulate across all model calls:

```python
total_input_tokens = 0
total_output_tokens = 0

async for event in graph.astream_events(state, config, version="v2"):
    # ... handle streaming output ...

    if event["event"] == "on_chat_model_end":
        output = event["data"]["output"]
        usage = extract_usage(output)
        total_input_tokens += usage["input_tokens"]
        total_output_tokens += usage["output_tokens"]

print(f"\nTotal: {total_input_tokens} in, {total_output_tokens} out")
```

Note that prompt tokens increase with each model call because the context
window grows (each tool result is added to the conversation before the next
call). This is expected and correct behaviour.

---

## 9.9  Rich Console Integration

### 9.9.1  Why Rich?

Python's `print()` function is fine for plain text, but saathi targets a
developer audience that appreciates polished terminal UX. The `rich` library
provides:

- Automatic terminal width detection
- Theme-aware colour support (respects `NO_COLOR` and `TERM`)
- Markdown rendering (headings, bold, italic, code blocks)
- Progress spinners for long-running operations
- Panels, tables, and other structured output components

The key challenge with Rich and streaming is that Rich's Markdown renderer
requires complete text to work correctly. You cannot render partial Markdown
because you do not know whether a `#` is the start of a heading or a Python
comment until you see the surrounding context.

saathi's approach: stream raw text during generation, then re-render the
complete response as Markdown after generation completes.

### 9.9.2  saathi's `display.py` Helpers

```python
# src/saathi/ui/display.py

from __future__ import annotations
import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text
from rich.style import Style

# One shared console instance. Configured to write to stderr so that
# status messages and tool indicators do not pollute stdout output.
_status_console = Console(stderr=True, highlight=False)
_output_console = Console(highlight=False)


def stream_token(token: str) -> None:
    """
    Print a single token during streaming.

    Uses print() directly rather than rich.Console to avoid Rich's
    buffering and markup processing during token-by-token output.
    """
    sys.stdout.write(token)
    sys.stdout.flush()


def show_tool_indicator(tool_name: str, status: str = "running") -> None:
    """
    Display a tool call indicator on stderr.

    Args:
        tool_name: The name of the tool being called.
        status: "running", "done", or "error".
    """
    style_map = {
        "running": Style(color="cyan", dim=True),
        "done": Style(color="green", dim=True),
        "error": Style(color="red"),
    }
    style = style_map.get(status, Style(dim=True))
    indicator = Text(f"  [{tool_name}: {status}]", style=style)
    _status_console.print(indicator)


def render_response(text: str) -> None:
    """
    Render a complete LLM response as Markdown.

    Called after streaming completes to re-render the response with
    proper syntax highlighting and Markdown formatting.
    """
    # Clear the streamed plain text (print newlines equivalent to the
    # text height, then move cursor up) — or simply trust that the
    # terminal will have already scrolled, and just render below.
    # In practice, saathi does not attempt to overwrite streamed text.
    # The Markdown render is only used for the history display.
    md = Markdown(text, code_theme="monokai")
    _output_console.print(md)


def show_separator() -> None:
    """Print a visual separator between turns."""
    _status_console.rule(style="dim")
```

### 9.9.3  Streaming with Rich's `Live` Context

For a more sophisticated streaming experience, Rich provides a `Live` context
manager that can update a region of the terminal in place. This allows you to
overwrite the streamed text with a properly formatted Markdown version:

```python
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

console = Console()

async def stream_with_live_markdown(graph, state, config):
    """
    Stream tokens, updating a live Markdown display as each token arrives.
    """
    accumulated_text = ""

    with Live(Markdown(""), console=console, refresh_per_second=15) as live:
        async for event in graph.astream_events(state, config, version="v2"):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if isinstance(chunk.content, str) and chunk.content:
                    accumulated_text += chunk.content
                    # Update the live display with the current accumulated text
                    live.update(Markdown(accumulated_text))
```

This approach re-renders the Markdown on every token. For responses up to a
few hundred tokens it is smooth. For very long responses (1000+ tokens), the
re-rendering cost can cause visible flicker. In production, saathi uses a
simpler approach: stream raw text, then optionally display formatted output
from the history on the next turn.

### 9.9.4  Disabling Rich for Non-Interactive Use

When saathi is used non-interactively (piped input/output, CI environments,
the `--print` flag), Rich's formatting is undesirable. ANSI escape codes
pollute the output, and terminal-width detection may fail.

saathi detects non-interactive contexts:

```python
import sys
import os

def is_interactive() -> bool:
    """Return True if stdin and stdout are connected to a TTY."""
    return sys.stdin.isatty() and sys.stdout.isatty()

def make_console(*, stderr: bool = False) -> Console:
    """Create a Console instance appropriate for the current context."""
    force_terminal = is_interactive()
    no_color = not force_terminal or os.environ.get("NO_COLOR", "") != ""
    return Console(
        stderr=stderr,
        force_terminal=force_terminal,
        no_color=no_color,
        highlight=False,
    )
```

Rich's `Console` respects the `NO_COLOR` environment variable automatically
when `no_color=True` is not explicitly set, but setting it explicitly ensures
consistent behaviour.

---

## 9.10  `--print` Mode and Streaming

### 9.10.1  The `--print` Flag

saathi's `--print` flag enables non-interactive, pipeline-friendly output.
In this mode:

- The response text is printed to stdout (no colours, no Rich formatting).
- Status messages, tool indicators, and token usage go to stderr.
- The prompt is read from a command-line argument or stdin.
- The process exits after a single response.

Example usage:

```bash
# Pipe the response to another command
saathi --print "Summarise the changes in this diff" < changes.diff | pbcopy

# Use in a shell script
SUMMARY=$(saathi --print "What does this function do?" < mycode.py)
echo "Summary: $SUMMARY"
```

### 9.10.2  Why Stream Internally in `--print` Mode?

A natural question: if `--print` only emits the final response, why bother
with streaming at all? Why not just use `ainvoke`?

Two reasons:

1. **Progress visibility**: Even in non-interactive contexts, it is useful to
   see that the model is making progress. Streaming allows saathi to print
   incremental status to stderr so that a script's author can see the model
   working rather than wondering if the process is hung.

2. **Long responses**: For very long responses (multi-page explanations,
   entire file generation), streaming allows the stdout buffer to be written
   incrementally. This means the consuming process (the one reading from the
   pipe) can start processing before saathi finishes generating.

### 9.10.3  The `_print_mode()` Pattern

```python
# src/saathi/cli.py (simplified)

import sys
import asyncio
from typing import AsyncIterator
from langchain_core.messages import AIMessageChunk


async def _print_mode(
    graph,
    state: dict,
    config: dict,
) -> str:
    """
    Run the graph in print mode: collect all tokens, write to stdout,
    status messages to stderr.

    Returns the complete response text.
    """
    accumulated_parts: list[str] = []

    async for event in graph.astream_events(state, config, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk: AIMessageChunk = event["data"]["chunk"]
            if isinstance(chunk.content, str) and chunk.content:
                accumulated_parts.append(chunk.content)

        elif kind == "on_tool_start":
            # Status to stderr so it does not pollute stdout
            print(f"[tool: {event['name']}]", file=sys.stderr, flush=True)

        elif kind == "on_tool_end":
            print(f"[tool: {event['name']}: done]", file=sys.stderr, flush=True)

        elif kind == "on_chat_model_end":
            output = event["data"]["output"]
            usage = extract_usage(output)
            if usage["total_tokens"] > 0:
                print(format_usage(usage), file=sys.stderr)

    full_response = "".join(accumulated_parts)

    # Write the complete response to stdout
    sys.stdout.write(full_response)
    if not full_response.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()

    return full_response
```

The function collects all tokens internally and writes the complete response
at the end. Alternatively, you can write tokens to stdout as they arrive
(for pipeline processing), which requires the consuming process to handle
incomplete lines gracefully.

### 9.10.4  Detecting `--print` Mode

```python
import argparse

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="saathi — AI assistant CLI")
    parser.add_argument(
        "--print",
        dest="print_mode",
        action="store_true",
        help="Non-interactive mode: print response to stdout and exit.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt to send in --print mode. If omitted, reads from stdin.",
    )
    return parser


async def main() -> None:
    args = build_parser().parse_args()

    if args.print_mode:
        prompt = args.prompt or sys.stdin.read().strip()
        if not prompt:
            print("Error: no prompt provided", file=sys.stderr)
            sys.exit(1)
        await _print_mode(graph, build_state(prompt), config)
    else:
        await _interactive_loop(graph, config)
```

---

## 9.11  Interrupting Streams

### 9.11.1  Ctrl+C During Generation

When the user presses Ctrl+C during streaming, Python raises
`KeyboardInterrupt`. In an async context (inside `asyncio.run()`), this
translates to the event loop being interrupted. The async generator
`astream_events` will raise `asyncio.CancelledError` when its underlying
tasks are cancelled.

Both exceptions must be handled correctly:

1. Do not silently swallow them — allow clean shutdown.
2. Print a newline so the terminal cursor is not left mid-line.
3. Do not leave the graph in an inconsistent state (LangGraph's checkpointing
   handles this automatically).

### 9.11.2  Basic Interrupt Handling

```python
async def stream_with_interrupt_handling(graph, state, config) -> str | None:
    """
    Stream graph output, handling Ctrl+C gracefully.

    Returns the partial response text, or None if interrupted before
    any output.
    """
    parts: list[str] = []

    try:
        async for event in graph.astream_events(state, config, version="v2"):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if isinstance(chunk.content, str) and chunk.content:
                    parts.append(chunk.content)
                    print(chunk.content, end="", flush=True)

    except KeyboardInterrupt:
        # User pressed Ctrl+C
        print("\n[interrupted]", file=sys.stderr)
    except asyncio.CancelledError:
        # asyncio task was cancelled (e.g., from signal handler)
        print("\n[cancelled]", file=sys.stderr)
        raise  # Re-raise CancelledError so asyncio can shut down cleanly
    finally:
        # Always ensure we end on a new line
        print()

    return "".join(parts) if parts else None
```

Note that `asyncio.CancelledError` should generally be re-raised after
cleanup, unless you are at the top level of an `asyncio.run()` call. Swallowing
`CancelledError` can leave asyncio's task machinery in an undefined state.

### 9.11.3  Signal Handling for Graceful Shutdown

For a more robust approach, install a signal handler that sets a flag:

```python
import asyncio
import signal
import sys

# Global cancellation flag
_interrupted = False


def _install_interrupt_handler(loop: asyncio.AbstractEventLoop) -> None:
    """
    Install a SIGINT handler that cancels the current streaming task
    cleanly rather than raising KeyboardInterrupt mid-await.
    """
    def handle_sigint():
        global _interrupted
        _interrupted = True
        # Cancel all running tasks
        for task in asyncio.all_tasks(loop):
            task.cancel()

    loop.add_signal_handler(signal.SIGINT, handle_sigint)


async def stream_with_signal_handler(graph, state, config):
    loop = asyncio.get_event_loop()
    _install_interrupt_handler(loop)

    try:
        async for event in graph.astream_events(state, config, version="v2"):
            if _interrupted:
                break
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if isinstance(chunk.content, str) and chunk.content:
                    print(chunk.content, end="", flush=True)
    except asyncio.CancelledError:
        pass
    finally:
        print()
        if _interrupted:
            print("[generation stopped]", file=sys.stderr)
```

Windows does not support `loop.add_signal_handler()`. On Windows, saathi
falls back to catching `KeyboardInterrupt` in the outer `try` block.

### 9.11.4  Partial Response Preservation

When the user interrupts a stream, saathi saves the partial response to the
conversation history. This allows the user to:

- Review what was generated before the interrupt.
- Continue the conversation naturally ("keep going", "I meant to ask...").
- Understand where the model was in its reasoning.

```python
async def interactive_turn(graph, state, config) -> None:
    """Handle one user turn, preserving partial responses on interrupt."""
    partial_response = await stream_with_interrupt_handling(graph, state, config)

    if partial_response:
        # Add partial response to history even if interrupted
        # LangGraph's state management handles this
        pass  # The graph's checkpointing already saved the state
    else:
        print("[no response generated]", file=sys.stderr)
```

Because LangGraph checkpoints state after each graph step, even an interrupted
generation preserves the messages that were committed before the interrupt.

---

## 9.12  Streaming in Tests

### 9.12.1  The Challenge

Streaming makes unit testing harder for several reasons:

1. **Async generators are harder to mock than coroutines.** You need to create
   an async generator mock, not just a coroutine mock.

2. **Event structure must be correct.** Your streaming loop code depends on
   the exact structure of the event dictionary. Test mocks must replicate this
   structure faithfully.

3. **Timing effects.** In production, tokens arrive over time. In tests, they
   arrive instantly, which can expose race conditions or buffering bugs that
   do not appear in production.

4. **Side effects.** The streaming loop may call `print()`, update a spinner,
   or modify global state. These side effects need to be captured and verified.

### 9.12.2  saathi's Testing Strategy

saathi's unit tests use `ainvoke` rather than `astream_events` for most
tests. This is a deliberate trade-off: `ainvoke` returns the final state as a
single awaitable, which is much easier to mock and assert against.

```python
# tests/test_graph.py

import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import AIMessage, HumanMessage


@pytest.mark.asyncio
async def test_basic_response(mock_graph):
    """Test that the graph produces a correct response."""
    # Use ainvoke, not astream_events
    result = await mock_graph.ainvoke(
        {"messages": [HumanMessage(content="Hello")]},
        config={"configurable": {"thread_id": "test-1"}},
    )
    assert "messages" in result
    last_message = result["messages"][-1]
    assert isinstance(last_message, AIMessage)
    assert len(last_message.content) > 0
```

The streaming-specific behaviour (token ordering, chunk assembly, interrupt
handling) is tested through integration tests that run against a real or
mock Ollama server.

### 9.12.3  Mocking `astream_events`

When you do need to test streaming-specific code, use an async generator
function as the mock:

```python
from unittest.mock import patch, AsyncMock
from langchain_core.messages import AIMessageChunk


def make_stream_events(*tokens: str):
    """
    Create a mock async generator that yields on_chat_model_stream events.

    Args:
        tokens: Individual token strings to emit.

    Returns:
        An async generator function that yields event dicts.
    """
    async def _generator(input, config=None, *, version="v2", **kwargs):
        for token in tokens:
            yield {
                "event": "on_chat_model_stream",
                "name": "ChatOllama",
                "run_id": "test-run-id",
                "parent_run_id": None,
                "tags": [],
                "metadata": {
                    "langgraph_node": "agent",
                    "langgraph_step": 1,
                },
                "data": {
                    "chunk": AIMessageChunk(content=token)
                },
            }
        # Final event
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "run_id": "test-run-id",
            "parent_run_id": None,
            "tags": [],
            "metadata": {},
            "data": {"output": {"messages": [AIMessageChunk(content="".join(tokens))]}},
        }

    return _generator


@pytest.mark.asyncio
async def test_streaming_output(capsys, mock_graph):
    """Test that the streaming loop prints tokens correctly."""
    mock_graph.astream_events = make_stream_events("Hello", ", ", "world", "!")

    await _print_mode(mock_graph, {"messages": []}, {})

    captured = capsys.readouterr()
    assert captured.out == "Hello, world!\n"
```

### 9.12.4  Testing Interrupt Handling

Testing `KeyboardInterrupt` in async code requires careful setup:

```python
@pytest.mark.asyncio
async def test_interrupt_handling():
    """Test that KeyboardInterrupt is handled gracefully."""
    interrupt_after = 2

    async def interrupting_generator(input, config=None, *, version="v2", **kwargs):
        tokens = ["This", " is", " a", " test"]
        for i, token in enumerate(tokens):
            if i >= interrupt_after:
                raise KeyboardInterrupt()
            yield {
                "event": "on_chat_model_stream",
                "name": "ChatOllama",
                "run_id": "test",
                "parent_run_id": None,
                "tags": [],
                "metadata": {},
                "data": {"chunk": AIMessageChunk(content=token)},
            }

    mock_graph = AsyncMock()
    mock_graph.astream_events = interrupting_generator

    result = await stream_with_interrupt_handling(mock_graph, {}, {})
    # Should have partial result from first two tokens
    assert result == "This is"
```

### 9.12.5  Fake LLMs for Integration Tests

For higher-level integration tests, LangChain provides `FakeListChatModel`
which returns predetermined responses:

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel

def make_fake_llm(responses: list[str]) -> FakeListChatModel:
    """
    Create a fake LLM that returns predetermined responses.

    Note: FakeListChatModel does NOT stream — it returns complete
    responses from ainvoke(). For streaming tests, use make_stream_events().
    """
    return FakeListChatModel(responses=responses)
```

`FakeListChatModel` does not implement `astream()`, so it cannot be used
directly with `astream_events`. For streaming tests, you must either use the
`make_stream_events()` helper shown above, or use LangChain's
`FakeStreamingListLLM` (though this is a text LLM, not a chat model).

The saathi test suite's philosophy: test business logic with `ainvoke` using
fake LLMs, and test streaming display code separately with mock generators.
This keeps both test types fast and deterministic.

---

## 9.13  `astream` (State Streaming)

### 9.13.1  What Is `astream`?

`graph.astream()` is a different streaming API from `astream_events()`. Where
`astream_events()` yields *lifecycle events* (token by token, tool by tool),
`astream()` yields the *graph state* after each node completes.

```python
async for state_update in graph.astream(input, config):
    # state_update is a dict of {node_name: node_output}
    print(state_update)
```

Each yielded value is a dictionary where the keys are node names and the
values are the outputs (state updates) produced by those nodes.

### 9.13.2  `astream` vs `astream_events`

| Feature                       | `astream`              | `astream_events`              |
| ------------------------------- | ------------------------ | ------------------------------- |
| Granularity                   | Per node               | Per token / per lifecycle event |
| Token streaming               | No                     | Yes                            |
| Tool call visibility          | After completion       | During execution               |
| State inspection              | Full state after node  | Partial (via events)           |
| Debugging utility             | High                   | Very high                      |
| Production use for printing   | No                     | Yes                            |
| Verbosity                     | Low                    | High                           |

For printing tokens to the user, `astream_events` is the right choice. For
debugging graph execution, `astream` is often more useful because it shows
the complete state transformation at each step.

### 9.13.3  Using `astream` for Debugging

```python
async def debug_graph_execution(graph, input_state, config):
    """
    Run the graph and print the full state after each node.
    Useful for understanding graph execution flow.
    """
    print("=== Graph Execution Debug ===\n")

    async for chunk in graph.astream(input_state, config):
        for node_name, node_output in chunk.items():
            print(f"Node: {node_name}")
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    msg_type = type(msg).__name__
                    content_preview = str(msg.content)[:100]
                    print(f"  {msg_type}: {content_preview!r}")
            print()
```

This produces output like:

```text
=== Graph Execution Debug ===

Node: agent
  AIMessage: 'Let me search for that information.'

Node: tools
  ToolMessage: '{"results": [{"title": "LangGraph", "url": "..."}]}'

Node: agent
  AIMessage: 'Based on the search results, LangGraph is a library...'
```

### 9.13.4  `astream` with `stream_mode`

LangGraph's `astream()` accepts a `stream_mode` parameter that controls what
is yielded:

- `"values"` (default): yields the full graph state after each node.
- `"updates"`: yields only the state *changes* (delta) from each node.
- `"debug"`: yields detailed debug information including task metadata.

```python
# Full state after each node (can be large)
async for state in graph.astream(input, config, stream_mode="values"):
    messages = state.get("messages", [])
    print(f"State has {len(messages)} messages")

# Only state changes from each node (more efficient)
async for update in graph.astream(input, config, stream_mode="updates"):
    for node_name, changes in update.items():
        print(f"{node_name} added: {changes}")
```

For large graphs with complex state, `stream_mode="updates"` is much more
efficient because it avoids copying the entire state on every node completion.

### 9.13.5  Combining `astream` and `astream_events`

In some advanced scenarios, you might want both state-level visibility
(from `astream`) and token-level visibility (from `astream_events`). These
cannot be combined in a single call, but you can run them sequentially:

1. During interactive use: use `astream_events` for token streaming.
2. For debugging a specific query: re-run with `astream` to inspect state.

Alternatively, the `on_chain_end` events in `astream_events` carry the node
output in `event["data"]["output"]`, giving you partial state visibility
without a separate `astream` call.

### 9.13.6  `astream` for Long-Running Graphs

For graphs with many steps or slow tools, `astream` can be used to display
progress:

```python
from rich.progress import Progress, SpinnerColumn, TextColumn

async def stream_with_progress(graph, input_state, config):
    """Show a progress spinner with the current node name."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Starting...", total=None)

        async for chunk in graph.astream(input_state, config, stream_mode="updates"):
            for node_name in chunk:
                progress.update(task, description=f"Running: {node_name}")
```

This gives users feedback on which part of the graph is executing, which is
particularly useful for multi-agent workflows with distinct planning,
execution, and summarisation phases.

---

## 9.14  Summary

Streaming is not a cosmetic feature — it is a fundamental part of how saathi
delivers a high-quality user experience. This chapter has traced the complete
path from the raw physics of LLM token generation through to the terminal
characters the user sees.

**The streaming pipeline in saathi:**

```flow
┌─────────────────────────────────────────────────────────────────┐
│  Ollama (NDJSON over HTTP)                                       │
│  {"content": "Hello", "done": false}                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  LangChain ChatOllama                                            │
│  AIMessageChunk(content="Hello")                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  LangGraph astream_events()                                     │
│  {"event": "on_chat_model_stream", "data": {"chunk": ...}}      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  saathi cli.py streaming loop                                    │
│  print(chunk.content, end="", flush=True)                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Terminal (user sees tokens as they are generated)              │
└─────────────────────────────────────────────────────────────────┘
```

Each layer of this pipeline transforms the data and adds context, while
preserving the essential property that tokens flow downstream without waiting
for the complete response.

**The key design decisions in saathi's streaming implementation:**

1. **Use `astream_events` with `version="v2"`** to get rich metadata including
   which LangGraph node produced each token.

2. **Write to `sys.stdout` directly** with `flush=True` for token output,
   bypassing Rich's buffering.

3. **Send status messages and metadata to stderr** so that stdout contains
   only the LLM's response text.

4. **Handle interrupts gracefully** by catching both `KeyboardInterrupt` and
   `asyncio.CancelledError`, printing a newline, and preserving any partial
   response.

5. **Track tool calls via `on_tool_start`/`on_tool_end`** rather than by
   trying to parse `tool_call_chunks`, since LangGraph handles tool execution.

6. **Use `ainvoke` in unit tests** and mock `astream_events` only when testing
   streaming-specific display logic.

7. **Centralise token usage extraction** in `usage.py` to handle both the
   standard `usage_metadata` field and Ollama-specific fallback fields.

---

## 9.15  Key Takeaways

- **Streaming dramatically improves perceived latency.** The user sees the
  first token within a second rather than waiting for the complete response.
  The total generation time is identical; only the experience changes.

- **`astream_events(version="v2")` is the primary streaming API for
  LangGraph.** It yields dictionaries with `event`, `name`, `data`, `run_id`,
  and `metadata` fields. The `langgraph_node` metadata field tells you which
  graph node produced each event.

- **`on_chat_model_stream` is the event type for token chunks.** Each event
  carries an `AIMessageChunk` in `event["data"]["chunk"]`. The `content` field
  holds the text token.

- **Tool calls also arrive as streamed chunks** via `tool_call_chunks` on the
  `AIMessageChunk`. Use `on_tool_start` for higher-level tool visibility.

- **`version="v2"` adds LangGraph node metadata** that is absent in v1. Use
  v2 for all new LangGraph applications.

- **`print(chunk.content, end="", flush=True)` is the minimum viable
  streaming output.** Always use `flush=True` to ensure tokens appear
  immediately.

- **Token usage is available from `on_chat_model_end`** via
  `event["data"]["output"].usage_metadata`. For Ollama, fall back to
  `response_metadata["prompt_eval_count"]` and `response_metadata["eval_count"]`.

- **In `--print` mode, stream internally but write to stdout atomically.**
  Status messages go to stderr. This enables pipeline use without polluting
  the output with ANSI codes or incomplete lines.

- **`KeyboardInterrupt` and `asyncio.CancelledError` must both be handled**
  during streaming. Always ensure a final newline is printed before the
  exception propagates.

- **Unit tests should use `ainvoke` for business logic** and mock
  `astream_events` only when testing display-specific streaming code.
  `FakeListChatModel` does not stream; use custom async generator mocks for
  streaming tests.

- **`astream` (state streaming) complements `astream_events` (event streaming)**
  by yielding the full graph state after each node. Use `astream` for debugging
  and `astream_events` for user-facing output.

- **The Rich library's `Live` context** enables in-place Markdown rendering
  during streaming, but comes with re-rendering overhead. For most cases,
  streaming raw text and rendering Markdown from history is more practical.

---

## Appendix: Streaming Cheat Sheet

### Event Types Quick Reference

```python
# Token output
if event["event"] == "on_chat_model_stream":
    token = event["data"]["chunk"].content  # str

# Model finished
if event["event"] == "on_chat_model_end":
    message = event["data"]["output"]  # AIMessage
    usage = message.usage_metadata  # {"input_tokens": N, "output_tokens": M}

# Tool starting
if event["event"] == "on_tool_start":
    tool_name = event["name"]  # str
    tool_input = event["data"]["input"]  # dict

# Tool finished
if event["event"] == "on_tool_end":
    tool_name = event["name"]  # str
    tool_output = event["data"]["output"]  # str or dict

# Node starting/ending (chain = graph node in LangGraph)
if event["event"] == "on_chain_start":
    node_name = event["name"]  # str

if event["event"] == "on_chain_end":
    node_name = event["name"]  # str
    output = event["data"]["output"]  # node output state
```

### Minimal Production Streaming Loop

```python
import asyncio
import sys


async def run(graph, state, config):
    try:
        async for event in graph.astream_events(state, config, version="v2"):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if isinstance(chunk.content, str) and chunk.content:
                    sys.stdout.write(chunk.content)
                    sys.stdout.flush()
            elif event["event"] == "on_tool_start":
                sys.stderr.write(f"\n[{event['name']}...]\n")
    except KeyboardInterrupt:
        sys.stderr.write("\n[interrupted]\n")
    finally:
        sys.stdout.write("\n")
        sys.stdout.flush()
```

### Key Imports

```python
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.tool import ToolCallChunk
from langgraph.graph import StateGraph, START, END
```

---

*Chapter 9 complete. The next chapter covers Memory and Persistence: how
saathi uses LangGraph's checkpointer to maintain conversation history across
sessions, and how to inspect, replay, and fork conversation threads.*
