# Chapter 3 — Ollama: Local LLM Serving for Production

This chapter covers Ollama from the ground up — what it is, how to install and configure it, how
saathi-langgraph talks to it, and how to keep the connection reliable in production. By the end you
will understand every knob that affects inference performance, why the retry logic in
`src/saathi/retry.py` is written the way it is, and how to run Ollama on a remote machine without
opening it to the public internet.

---

## 1. What is Ollama?

Ollama is a single-binary daemon that downloads, manages, and serves large language models through a
clean REST API. Under the hood it is a Go application that shells out to **llama.cpp** for the
actual matrix math, but that implementation detail is almost entirely hidden. You interact with
Ollama through HTTP, the same way you would interact with the OpenAI API — except everything runs
on your own hardware and nothing leaves your machine.

### The llama.cpp Connection

llama.cpp is a C++ library that can run quantised GGUF model files on commodity hardware. It
supports every acceleration back-end that matters today:

- **CUDA** — NVIDIA GPUs via the CUDA toolkit.
- **ROCm / HIP** — AMD GPUs on Linux.
- **Metal** — Apple Silicon (M1 / M2 / M3 / M4) via the Metal GPU API.
- **Vulkan** — cross-platform GPU compute; useful on Windows with non-NVIDIA cards.
- **BLAS** — CPU-only inference with OpenBLAS or Apple's Accelerate framework.

llama.cpp detects the available back-end at build time. Ollama ships with pre-built llama.cpp
binaries for each back-end and selects the right one at runtime based on what hardware it finds.
You never have to compile anything.

### How Ollama Differs from Running llama.cpp Directly

Running llama.cpp directly means invoking the `main` executable (or the `llama-cli` binary in
newer builds) from the command line. This works for one-off experiments but falls short for
anything resembling a real application:

| Capability | llama.cpp directly | Ollama |
| --- | --- | --- |
| HTTP API | Manual (`llama-server`) | Built-in |
| Model registry | None — you manage files | Pull by name |
| Multi-model management | One process per model | Daemon manages all |
| Automatic GPU detection | Compile-time flags | Runtime detection |
| Keep-alive / model caching | None | OLLAMA_KEEP_ALIVE |
| Concurrent request queuing | None | Built-in queue |
| System prompt storage | Command-line flag | Modelfile |

Ollama wraps `llama-server` (the HTTP mode of llama.cpp) inside a process manager that also
handles model downloads, GGUF file storage, and a unified REST interface. From the application's
point of view there is exactly one base URL and one token to worry about.

### Architecture Overview

```text
┌────────────────────────────────────────────────────────────────────┐
│                        saathi-langgraph                            │
│                                                                    │
│   ┌─────────────┐    LangChain      ┌──────────────────────────┐  │
│   │  CLI (cli.py)│ ──────────────▶  │  ChatOllama (LangChain)  │  │
│   └─────────────┘                   └──────────┬───────────────┘  │
│                                                │ HTTP POST         │
└────────────────────────────────────────────────┼───────────────────┘
                                                 │
                              ┌──────────────────▼───────────────────┐
                              │              Ollama Daemon            │
                              │                                       │
                              │  ┌─────────────┐  ┌──────────────┐  │
                              │  │ Model Cache  │  │  llama-server │  │
                              │  │ (~/.ollama)  │  │  (llama.cpp)  │  │
                              │  └──────┬───────┘  └──────┬───────┘  │
                              │         │  loads           │ infers   │
                              │         └──────────────────┘         │
                              │                  │                    │
                              │         ┌────────▼────────┐          │
                              │         │  GPU / CPU VRAM  │          │
                              │         │  (CUDA / Metal)  │          │
                              │         └─────────────────┘          │
                              └────────────────────────────────────────┘
```

The daemon listens on `localhost:11434` by default. Every request goes to this single endpoint.
Ollama loads the model into GPU VRAM (or RAM if you have no GPU) the first time a model is
requested and keeps it resident for subsequent requests, avoiding the multi-second cold-start cost
on every call.

### Why saathi Uses Ollama

saathi-langgraph is designed to run entirely offline. Its users are developers who want an AI pair
programmer that does not send their proprietary source code to a remote server. Ollama satisfies
this requirement while providing a LangChain-compatible interface through `ChatOllama`. The project
also needs reliable connection handling, which is why `src/saathi/retry.py` treats the Ollama
connection with the same care you would give any external service dependency.

---

## 2. Installing and Starting Ollama

Ollama provides native installers for all three major platforms. The installation path and the
mechanism for auto-starting the daemon differ by platform; the HTTP API is identical everywhere.

### Linux

The official one-liner downloads and runs an install script that copies the Ollama binary to
`/usr/local/bin` and installs a systemd unit file:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

After installation the daemon starts automatically via systemd. To check the service status:

```bash
systemctl status ollama
```

The systemd unit file lives at `/etc/systemd/system/ollama.service`. A typical unit file looks
like this:

```ini
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

[Install]
WantedBy=default.target
```

If you need to customise environment variables — for example to set `OLLAMA_HOST` or
`OLLAMA_NUM_PARALLEL` — create a drop-in override rather than editing the unit file directly:

```bash
sudo systemctl edit ollama
```

This opens an editor where you can add:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_NUM_PARALLEL=2"
Environment="OLLAMA_KEEP_ALIVE=30m"
```

After saving, reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

To start Ollama manually (without systemd, useful for debugging):

```bash
OLLAMA_HOST=127.0.0.1:11434 ollama serve
```

### macOS

Download the `.dmg` from [https://ollama.com/download](https://ollama.com/download), open it, and
drag Ollama to the Applications folder. Double-clicking the application launches the daemon and adds
a menu-bar icon. Ollama registers itself as a Login Item so it restarts after reboot.

On Apple Silicon Macs the Metal back-end is used automatically. You can verify this by looking at
the Ollama log output, which will mention `ggml_metal_init`.

To set environment variables on macOS, create a launchd plist override. For development work it is
usually simpler to set them in the shell before running `ollama serve` directly:

```bash
OLLAMA_NUM_PARALLEL=2 OLLAMA_KEEP_ALIVE=1h ollama serve
```

### Windows

Download and run the `.exe` installer from [https://ollama.com/download](https://ollama.com/download).
The installer adds Ollama to the system tray and configures it to start with Windows. CUDA
acceleration works on NVIDIA cards if the CUDA toolkit is installed; Vulkan is used as a fallback
for other GPU brands.

To set environment variables on Windows, use the System Properties dialog or PowerShell:

```powershell
[System.Environment]::SetEnvironmentVariable(
    "OLLAMA_NUM_PARALLEL", "2",
    [System.EnvironmentVariableTarget]::User
)
```

Then restart the Ollama service from the system tray.

### The OLLAMA_HOST Environment Variable

`OLLAMA_HOST` controls both which interface the daemon listens on and which address client commands
use when you run `ollama pull`, `ollama list`, and so on. The format is `HOST:PORT`.

```bash
# Listen on all interfaces (needed for remote access)
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# Listen on a non-default port
OLLAMA_HOST=127.0.0.1:11435 ollama serve

# Tell the CLI to connect to a remote Ollama instance
OLLAMA_HOST=192.168.1.42:11434 ollama list
```

The client tools (`ollama pull`, `ollama list`, etc.) read `OLLAMA_HOST` from the environment, so
you can point them at a remote instance without changing any configuration files.

> **Note:** When `OLLAMA_HOST` is set for the daemon, you must also set it for the client commands
> if you are running them in a different shell. They default to `localhost:11434` regardless of what
> the daemon is doing.

### Verifying the Installation

After starting the daemon, confirm it is responding:

```bash
curl http://localhost:11434/api/tags
```

A healthy response is a JSON object with a `"models"` array (empty if you have not pulled any
models yet):

```json
{"models":[]}
```

---

## 3. The Ollama HTTP API

Ollama exposes a REST API on port 11434. The API is small and well-designed. Understanding it
directly — without LangChain in the way — makes debugging much easier when something goes wrong.

### POST /api/generate

This is the completion endpoint. It takes a model name and a prompt and returns generated text.

```bash
curl http://localhost:11434/api/generate \
  -d '{
    "model": "gemma3:12b",
    "prompt": "What is the difference between a list and a tuple in Python?",
    "stream": false
  }'
```

With `"stream": false` the response is a single JSON object:

```json
{
  "model": "gemma3:12b",
  "created_at": "2026-07-09T10:00:00.000Z",
  "response": "A list is mutable while a tuple is immutable...",
  "done": true,
  "context": [1, 2, 3],
  "total_duration": 4200000000,
  "load_duration": 500000,
  "prompt_eval_count": 18,
  "eval_count": 132,
  "eval_duration": 3800000000
}
```

The timing fields are in nanoseconds. `eval_count / (eval_duration / 1e9)` gives tokens per second.

With `"stream": true` (the default) the response is a stream of newline-delimited JSON objects,
one per generated token or small token batch:

```json
{"model":"gemma3:12b","created_at":"2026-07-09T10:00:00.000Z","response":"A","done":false}
{"model":"gemma3:12b","created_at":"2026-07-09T10:00:00.001Z","response":" list","done":false}
...
{"model":"gemma3:12b","created_at":"2026-07-09T10:00:04.200Z","response":"","done":true,"eval_count":132}
```

### POST /api/chat

This is the chat completion endpoint and is what `ChatOllama` in LangChain uses. It takes a list of
messages rather than a raw prompt string.

```bash
curl http://localhost:11434/api/chat \
  -d '{
    "model": "gemma3:12b",
    "messages": [
      {"role": "system", "content": "You are a helpful Python tutor."},
      {"role": "user", "content": "Explain list comprehensions."}
    ],
    "stream": false
  }'
```

Response structure:

```json
{
  "model": "gemma3:12b",
  "created_at": "2026-07-09T10:00:00.000Z",
  "message": {
    "role": "assistant",
    "content": "List comprehensions provide a concise way to create lists..."
  },
  "done": true,
  "total_duration": 3900000000,
  "eval_count": 98
}
```

When tool calling is enabled and the model generates a tool call, the `message` object gains a
`tool_calls` field instead of (or alongside) `content`. This is covered in detail in Section 7.

### GET /api/tags

Lists all locally available models. No request body needed.

```bash
curl http://localhost:11434/api/tags
```

```json
{
  "models": [
    {
      "name": "gemma3:12b",
      "modified_at": "2026-07-01T12:00:00.000Z",
      "size": 8070044160,
      "digest": "sha256:abc123...",
      "details": {
        "format": "gguf",
        "family": "gemma3",
        "parameter_size": "12B",
        "quantization_level": "Q4_K_M"
      }
    }
  ]
}
```

This endpoint is what `src/saathi/diagnostics.py` calls to check that Ollama is running and that
the configured model is available before the user starts a conversation.

### POST /api/pull

Pulls a model from the Ollama registry. This is what `ollama pull` calls internally.

```bash
curl http://localhost:11434/api/pull \
  -d '{"name": "gemma3:12b", "stream": false}'
```

With streaming enabled (the default) the response is a stream of progress events:

```json
{"status":"pulling manifest"}
{"status":"pulling 8ab4849b038c","digest":"sha256:8ab4849b038c...","total":8070044160,"completed":1048576}
...
{"status":"success"}
```

### Parsing the Streaming Response in Python

Both `/api/generate` and `/api/chat` stream newline-delimited JSON by default. Here is a minimal
Python function that reads the stream and prints tokens as they arrive:

```python
import httpx
import json


def stream_completion(model: str, prompt: str, base_url: str = "http://localhost:11434") -> str:
    """Stream a completion from Ollama, printing each token, returning the full text."""
    collected = []

    with httpx.Client() as client:
        with client.stream(
            "POST",
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": True},
            timeout=None,  # No timeout for streaming inference
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                print(token, end="", flush=True)
                collected.append(token)
                if chunk.get("done"):
                    break

    print()  # Final newline
    return "".join(collected)
```

> **Note:** The `timeout=None` on the streaming call is intentional. Once the connection is
> established and the model starts generating, individual tokens can take a long time on slow
> hardware. A read timeout here would incorrectly cancel an in-progress generation. See Section 9
> for a detailed discussion of this design decision.

---

## 4. Model Management

Ollama stores models in `~/.ollama/models` (Linux and macOS) or
`%USERPROFILE%\.ollama\models` (Windows). Each model is stored as a set of GGUF blob files with a
manifest that describes them. You interact with models through the `ollama` CLI.

### Pulling Models

```bash
# Pull the default (latest) tag
ollama pull gemma3:12b

# Pull a specific quantisation
ollama pull gemma3:12b-it-q4_K_M

# Pull a code-focused model
ollama pull qwen2.5-coder:7b

# Pull the latest Llama 4 Scout variant
ollama pull llama4:scout
```

Tag names follow the format `model:variant`. When you omit the tag, Ollama pulls `latest`, which is
the recommended variant for that model family.

### Listing Models

```bash
ollama list
```

Output:

```text
NAME                    ID              SIZE    MODIFIED
gemma3:12b              8ab4849b038c    7.7 GB  2 weeks ago
qwen2.5-coder:7b        f6897c04bb9b    4.7 GB  3 days ago
llama4:scout            1234abcd5678    9.8 GB  1 day ago
```

### Removing Models

```bash
ollama rm gemma3:12b
```

This deletes the blob files and manifest. It does not affect other models that share blobs (Ollama
uses content-addressed storage; blobs are shared across models when possible).

### Running a Model Interactively

```bash
ollama run gemma3:12b
```

This drops you into an interactive chat session. Useful for quick experiments before integrating
a model into saathi.

```bash
# Pass a single prompt non-interactively
ollama run gemma3:12b "Explain the GIL in one paragraph."

# Pipe stdin
echo "What is LangGraph?" | ollama run gemma3:12b
```

### Modelfiles

A Modelfile is a declarative recipe for creating a custom Ollama model. It is similar to a
Dockerfile. The syntax supports:

```text
FROM <base-model>
SYSTEM <system-prompt>
PARAMETER <name> <value>
TEMPLATE <template-string>
MESSAGE <role> <content>
```

A simple example that wraps `gemma3:12b` with a coding-focused system prompt:

```dockerfile
FROM gemma3:12b

SYSTEM """
You are an expert Python and systems programmer. You give concise, correct answers
with working code examples. You always explain why, not just what. You point out
potential bugs and edge cases without being asked.
"""

PARAMETER temperature 0.2
PARAMETER num_ctx 8192
PARAMETER top_p 0.9
```

Build the custom model from this Modelfile:

```bash
ollama create saathi-coder -f ./Modelfile
```

The new model appears in `ollama list` as `saathi-coder:latest` and can be used anywhere that
accepts a model name. Section 14 covers Modelfiles in full.

### Showing Model Information

```bash
ollama show gemma3:12b
```

This prints the model's Modelfile (including the template and any default parameters), its
architecture details, and the quantisation format.

---

## 5. ChatOllama from LangChain

LangChain's `ChatOllama` class is the bridge between the LangGraph agent graph and the Ollama
daemon. It wraps the `/api/chat` endpoint and translates between LangChain's message types
(`HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`) and Ollama's JSON format.

### Constructor Parameters

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemma3:12b",          # Model name as shown in `ollama list`
    base_url="http://localhost:11434",  # Ollama daemon URL
    temperature=0.2,             # Sampling temperature (0.0 = greedy)
    num_ctx=8192,                # Context window in tokens
    num_predict=4096,            # Max tokens to generate (-1 = unlimited)
    top_p=0.9,                   # Nucleus sampling threshold
    top_k=40,                    # Top-K sampling
    repeat_penalty=1.1,          # Penalise repeated tokens
    timeout=120,                 # HTTP request timeout in seconds
    keep_alive="10m",            # How long to keep model in VRAM after last use
)
```

The most important parameters for saathi are:

- **`model`** — matches `SAATHI_MODEL` from the environment, defaulting to `gemma3:12b`.
- **`num_ctx`** — controls how much of the conversation history and file context fits in the model's
  attention window. Larger values give better context at the cost of more VRAM and slower inference.
- **`temperature`** — lower values make the model more deterministic and better for code generation.
- **`timeout`** — this is the *connection* timeout, not a read timeout. See Section 9.

### The make_llm() Function

In `src/saathi/agent/graph.py`, saathi constructs the LLM object through a factory function that
reads configuration from the environment:

```python
import os
from langchain_ollama import ChatOllama


def make_llm(
    model: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    num_ctx: int | None = None,
) -> ChatOllama:
    """
    Construct a ChatOllama instance from environment variables with
    overrides from keyword arguments.

    Environment variables:
        SAATHI_MODEL       — model name (default: gemma3:12b)
        SAATHI_BASE_URL    — Ollama base URL (default: http://localhost:11434)
        SAATHI_TEMPERATURE — sampling temperature (default: 0.2)
        SAATHI_NUM_CTX     — context window size in tokens (default: 8192)
    """
    resolved_model = model or os.getenv("SAATHI_MODEL", "gemma3:12b")
    resolved_base_url = base_url or os.getenv("SAATHI_BASE_URL", "http://localhost:11434")
    resolved_temperature = temperature if temperature is not None else float(
        os.getenv("SAATHI_TEMPERATURE", "0.2")
    )
    resolved_num_ctx = num_ctx or int(os.getenv("SAATHI_NUM_CTX", "8192"))

    return ChatOllama(
        model=resolved_model,
        base_url=resolved_base_url,
        temperature=resolved_temperature,
        num_ctx=resolved_num_ctx,
        num_predict=int(os.getenv("SAATHI_NUM_PREDICT", "4096")),
        keep_alive=os.getenv("SAATHI_KEEP_ALIVE", "10m"),
        # Connection timeout only — read timeout must be None for streaming
        timeout=int(os.getenv("SAATHI_CONNECT_TIMEOUT", "30")),
    )
```

> **Note:** The function signature uses `str | None` union syntax, which requires Python 3.10 or
> later. If you need to support earlier versions, use `Optional[str]` from `typing`.

### Invoking the LLM

The simplest invocation pattern is synchronous:

```python
from langchain_core.messages import HumanMessage, SystemMessage

llm = make_llm()

messages = [
    SystemMessage(content="You are a helpful Python assistant."),
    HumanMessage(content="What does the walrus operator do?"),
]

response = llm.invoke(messages)
print(response.content)
```

In practice the graph in `src/saathi/agent/graph.py` calls the LLM asynchronously through
LangGraph's node mechanism, which is covered in Chapter 4.

### Checking Model Availability Before Invocation

Before building the graph, saathi checks that the model exists locally:

```python
import httpx


def model_is_available(model: str, base_url: str = "http://localhost:11434") -> bool:
    """Return True if `model` appears in the local Ollama model list."""
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        response.raise_for_status()
        tags = response.json()
        available = [m["name"] for m in tags.get("models", [])]
        # Normalise: gemma3:12b and gemma3:12b-it-q4_K_M should both match "gemma3:12b"
        return any(name.startswith(model.split(":")[0]) for name in available)
    except (httpx.ConnectError, httpx.TimeoutException):
        return False
```

---

## 6. Streaming with ChatOllama

Streaming is essential for a good CLI experience. Without it the user stares at a blank screen for
several seconds until the entire response is ready. With streaming, tokens appear as they are
generated, providing immediate feedback that the model is working.

### How Streaming Works

When you call `.astream()` on a `ChatOllama` instance, LangChain opens a persistent HTTP connection
to `/api/chat` with `"stream": true` in the request body. Ollama writes one JSON object per line
as it generates each token. LangChain parses each line and yields an `AIMessageChunk` object with
the `content` field set to the newly generated text.

The chunks are small — often a single word or subword token. You assemble them by concatenating
their `content` fields. LangChain's `+` operator on message chunks handles this:

```python
from langchain_core.messages import AIMessageChunk

# Accumulate chunks
full: AIMessageChunk | None = None
async for chunk in llm.astream(messages):
    if full is None:
        full = chunk
    else:
        full = full + chunk

# full.content is now the complete response string
```

### Complete Async Streaming Example with Rich

The following example shows how saathi streams responses to the terminal using the `rich` library's
live display:

```python
import asyncio
from langchain_core.messages import HumanMessage, SystemMessage, AIMessageChunk
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown


async def stream_to_terminal(
    question: str,
    model: str = "gemma3:12b",
    base_url: str = "http://localhost:11434",
) -> str:
    """Stream an LLM response to the terminal using Rich live display."""
    llm = ChatOllama(model=model, base_url=base_url, temperature=0.2)
    console = Console()

    messages = [
        SystemMessage(content="You are a concise technical assistant."),
        HumanMessage(content=question),
    ]

    full_text = ""

    with Live(console=console, refresh_per_second=20) as live:
        async for chunk in llm.astream(messages):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                full_text += chunk.content
                # Re-render the accumulated markdown on each chunk
                live.update(Markdown(full_text))

    return full_text


if __name__ == "__main__":
    answer = asyncio.run(
        stream_to_terminal("Explain Python's asyncio event loop in three paragraphs.")
    )
    print(f"\n[{len(answer)} chars received]")
```

### Using astream_events()

`astream_events()` is a higher-level API that yields typed event dictionaries rather than raw
chunks. This is useful in LangGraph agents where you want to distinguish between the LLM thinking
and a tool execution. Each event has a `"event"` key and a `"data"` key:

```python
async def stream_with_events(messages: list, llm: ChatOllama) -> str:
    """Stream using astream_events, handling both text and tool-call events."""
    full_text = ""

    async for event in llm.astream_events(messages, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk: AIMessageChunk = event["data"]["chunk"]
            if chunk.content:
                full_text += chunk.content
                print(chunk.content, end="", flush=True)

        elif kind == "on_chat_model_end":
            # Final event; event["data"]["output"] is the complete AIMessage
            print()  # newline after the streamed text

    return full_text
```

### Assembling the Final Message

In a LangGraph node the standard pattern is:

```python
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.graph import add_messages


async def call_model_node(state: dict, llm: ChatOllama) -> dict:
    """LangGraph node that calls the LLM and returns the new state."""
    messages = state["messages"]
    chunks: list[AIMessageChunk] = []

    async for chunk in llm.astream(messages):
        chunks.append(chunk)

    # Combine all chunks into a single AIMessage
    full_message: AIMessage = sum(chunks[1:], chunks[0])

    return {"messages": [full_message]}
```

> **Warning:** Do not pass an empty list to `sum()`. Always check that `chunks` is non-empty
> before combining. An empty response from the model (rare but possible) will raise an error here.
> A safe pattern is `full_message = chunks[0] if len(chunks) == 1 else sum(chunks[1:], chunks[0])`.

---

## 7. Tool Calling with Ollama

Tool calling (also called function calling) allows the model to emit a structured JSON object
requesting a tool execution rather than a free-form text response. The application inspects the
response, runs the requested tool, and feeds the result back to the model in the next turn. This is
the mechanism saathi uses for file reading, shell execution, and web search.

### Which Models Support Tool Calling

Not every model supports tool calling. As of mid-2026 the reliably capable options available
through Ollama are:

| Model | Tool Calling | Notes |
| --- | --- | --- |
| Llama 4 Scout / Maverick | Yes | Strong; recommended for saathi |
| Gemma 3 12B / 27B | Yes | Good for code tasks |
| Gemma 4 | Yes | Best quality in the Gemma line |
| Mistral 7B Instruct v0.3+ | Yes | Solid but smaller context |
| Qwen2.5 / Qwen2.5-Coder | Yes | Excellent for code |
| Phi-4 | Partial | Sometimes unreliable |
| Older Llama 2 / 3 variants | No | Use a newer model |

Always check the model's Ollama page for current tool calling support status.

### Binding Tools with LangChain

LangChain tools are Python functions decorated with `@tool`. The `.bind_tools()` method on
`ChatOllama` attaches them to the LLM so that their JSON schema is sent with each request:

```python
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def read_file(path: str) -> str:
    """Read the contents of a file and return them as a string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@tool
def list_directory(path: str) -> list[str]:
    """List files and directories at the given path."""
    import os
    return os.listdir(path)


# Bind tools to the LLM
llm = ChatOllama(model="gemma3:12b", temperature=0.0)
llm_with_tools = llm.bind_tools([read_file, list_directory])

# Now when you invoke the LLM it may return tool calls
from langchain_core.messages import HumanMessage

response = llm_with_tools.invoke([
    HumanMessage(content="What files are in the /tmp directory?")
])

if response.tool_calls:
    print("Model wants to call:", response.tool_calls)
else:
    print("Model answered directly:", response.content)
```

### The tool_calls Field Structure

When the model requests a tool call, `response.tool_calls` is a list of dictionaries with this
structure:

```python
[
    {
        "name": "list_directory",
        "args": {"path": "/tmp"},
        "id": "call_abc123",
        "type": "tool_call",
    }
]
```

- **`name`** — the Python function name registered with `@tool`.
- **`args`** — a dictionary of argument names to values, already parsed from JSON.
- **`id`** — a unique identifier used to match the tool result back to the call.

### Completing a Tool Call Round-Trip

```python
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


async def run_tool_loop(question: str) -> str:
    """Run a simple single-turn tool-calling loop."""
    llm = ChatOllama(model="llama4:scout", temperature=0.0)
    llm_with_tools = llm.bind_tools([add])

    messages = [HumanMessage(content=question)]

    # First call — model may request a tool
    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)

    # Execute any requested tools
    for tool_call in response.tool_calls:
        if tool_call["name"] == "add":
            result = add.invoke(tool_call["args"])
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

    # Second call — model has tool results, produces final answer
    if response.tool_calls:
        final = await llm_with_tools.ainvoke(messages)
        return final.content

    return response.content
```

### How Tool Call JSON Is Embedded in Ollama's Response

At the raw HTTP level, when a model makes a tool call the `/api/chat` response's `message` object
looks like this:

```json
{
  "role": "assistant",
  "content": "",
  "tool_calls": [
    {
      "function": {
        "name": "list_directory",
        "arguments": "{\"path\": \"/tmp\"}"
      }
    }
  ]
}
```

Note that `arguments` is a JSON-encoded string, not a nested object. LangChain's `ChatOllama`
implementation parses this and exposes it as the `args` dictionary shown earlier.

> **Warning:** Not all models reliably produce valid JSON in the `arguments` field. If the model
> is poorly quantised or the tool schema is very complex, it may generate malformed JSON. The
> LangGraph agent in saathi handles this by catching `json.JSONDecodeError` and retrying the call
> with a corrective message that instructs the model to fix its output.

---

## 8. Performance Tuning

Getting good performance from Ollama requires tuning a handful of variables that control
parallelism, memory retention, and inference parameters. The right values depend on your hardware,
but this section gives you a principled framework for finding them.

### OLLAMA_NUM_PARALLEL

Controls how many inference requests Ollama processes simultaneously. Each concurrent request
requires a separate KV cache allocation, which consumes VRAM.

```bash
# Allow 2 concurrent requests (default is 1)
OLLAMA_NUM_PARALLEL=2 ollama serve
```

With `OLLAMA_NUM_PARALLEL=1` (the default) requests are queued. The second caller waits until the
first finishes. For saathi's single-user CLI use case this is usually fine. If you run saathi as a
server with multiple clients, consider increasing this value — but watch your VRAM usage closely.

### OLLAMA_KEEP_ALIVE

After a request completes, Ollama keeps the model loaded in VRAM for this duration. If another
request arrives within the window, the model is already in memory (no load time). After the window
expires the model is evicted to free VRAM.

```bash
# Keep model loaded for 30 minutes (default is 5 minutes)
OLLAMA_KEEP_ALIVE=30m ollama serve

# Keep model loaded forever (useful if you have dedicated VRAM)
OLLAMA_KEEP_ALIVE=-1 ollama serve

# Unload immediately after each request (conserves VRAM)
OLLAMA_KEEP_ALIVE=0 ollama serve
```

For interactive use like saathi, a keep-alive of 10–30 minutes eliminates the cold-start delay
between conversation turns.

### num_ctx vs Performance

The context window (`num_ctx`) is the single biggest lever on VRAM usage and inference speed. The
KV cache size scales quadratically with sequence length, so a context of 32768 tokens uses roughly
4× as much VRAM as 8192 tokens.

| num_ctx | Approximate KV Cache (8B model, FP16) | Use Case |
| --- | --- | --- |
| 2048 | ~512 MB | Quick Q&A |
| 4096 | ~1 GB | Short coding sessions |
| 8192 | ~2 GB | Typical saathi session |
| 16384 | ~4 GB | Large file analysis |
| 32768 | ~8 GB | Repository-scale context |

Start with `num_ctx=8192` and increase only when you observe the model forgetting early parts of
the conversation.

### OLLAMA_FLASH_ATTENTION

Enables flash attention, a memory-efficient attention implementation that reduces VRAM usage at
long context lengths with no quality loss:

```bash
OLLAMA_FLASH_ATTENTION=1 ollama serve
```

This is particularly beneficial when `num_ctx` is above 8192. Enable it unconditionally on modern
hardware.

### Batch Size

The `num_batch` parameter controls how many tokens are processed in a single GPU kernel call during
the prefill phase (processing the prompt). Larger batches use more VRAM but reduce latency for
long prompts:

```python
llm = ChatOllama(
    model="gemma3:12b",
    num_batch=512,  # Default is 512; try 1024 on high-VRAM GPUs
)
```

### Performance Tuning Checklist

Work through this list when optimising a new deployment:

- [ ] Confirm GPU acceleration is active: check for `ggml_cuda_init` or `ggml_metal_init` in the
  Ollama log.
- [ ] Set `OLLAMA_FLASH_ATTENTION=1` if `num_ctx > 4096`.
- [ ] Set `OLLAMA_KEEP_ALIVE` to at least the expected inter-request interval.
- [ ] Choose the largest quantisation (Q8_0 or Q6_K) that fits in your VRAM.
- [ ] Measure baseline tokens/sec with `ollama run <model> "Write hello world"` and watch the
  throughput stat printed at the end.
- [ ] If VRAM is tight, reduce `num_ctx` before reducing the quantisation level — lower context
  preserves quality better than heavier quantisation at the same VRAM budget.
- [ ] For multi-user deployments, set `OLLAMA_NUM_PARALLEL` to the expected number of concurrent
  users, not higher.
- [ ] Check CPU offload: if layers are being offloaded to CPU (`llm_load_tensors: offloaded N/N
  layers`), reduce `num_ctx` or switch to a smaller model.

### Measuring Tokens Per Second

Ollama reports performance statistics in the final JSON object of each streaming response:

```python
import httpx, json

def measure_throughput(model: str, prompt: str) -> float:
    """Return tokens/sec for a single completion."""
    with httpx.Client() as client:
        lines = client.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": True},
            timeout=None,
        ).iter_lines()
        final = None
        for line in lines:
            final = json.loads(line)
    if final and final.get("eval_duration"):
        return final["eval_count"] / (final["eval_duration"] / 1e9)
    return 0.0
```

---

## 9. Connection Resilience

Ollama is a local daemon, not a managed cloud service. It can be down when saathi starts, it can
crash and restart, and a user's laptop can hibernate between conversation turns, ejecting the
daemon. saathi needs to handle all of these cases gracefully without losing the user's work.

### What Happens When Ollama is Starting Up

When the daemon is starting (or restarting after a crash), any HTTP request to port 11434 receives
a `Connection refused` error immediately. This is an `httpx.ConnectError`. The connection never
even opens — the kernel rejects it at the TCP layer.

This error is **safe to retry**. The daemon may be ready within a second or two of the first
failure. A retry with an exponential backoff will succeed once the daemon is fully initialised.

### Read Timeouts Are Different

A `httpx.ReadTimeout` occurs when the connection is successfully established and the request is
sent, but the server does not send any response bytes within the configured timeout window.

For Ollama this has a very specific meaning: the model is loaded, the prompt is being processed,
and the first token has not been generated yet. The model is *thinking*. This is not an error.

If you retry on `ReadTimeout`, you do the following:

1. Cancel the in-progress inference on the server side (by closing the connection).
2. Start a new inference with the same prompt.
3. Wait again.
4. Repeat until either the model responds quickly enough or you exhaust retries.

This turns a slow-but-correct response into an infinite loop that monopolises the GPU and never
produces output. It also makes the user's problem worse: each retry resets the wait time.

The correct response to `ReadTimeout` is to surface it to the user and let them decide whether to
cancel or wait longer. You might also consider not setting a read timeout at all for inference
requests (pass `timeout=None` or set `read=None` in the `httpx.Timeout` object).

### The retry.py Module

`src/saathi/retry.py` implements the retry logic for Ollama connections. The key design decision
is the `RETRYABLE` tuple, which explicitly includes connection errors but excludes read timeouts:

```python
"""
src/saathi/retry.py
-------------------
Retry helpers for Ollama HTTP connections.

Design note on RETRYABLE
------------------------
We retry only on errors that indicate the connection could not be
established. A ConnectError means Ollama is not yet listening; a
ConnectTimeout means DNS resolution or TCP handshake timed out.
Both are transient and safe to retry.

We deliberately EXCLUDE ReadTimeout. A ReadTimeout from Ollama means
the model is processing the prompt and has not generated its first
token yet. Retrying would:
  1. Cancel the current inference (by dropping the TCP connection).
  2. Restart inference from scratch.
  3. Potentially loop forever on a slow model / long prompt.

If a read timeout fires, the caller should surface it to the user
rather than silently retrying.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, TypeVar

import httpx

logger = logging.getLogger(__name__)

# Errors that indicate the daemon is not yet ready.
# ReadTimeout is intentionally absent — see module docstring.
RETRYABLE = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
)

T = TypeVar("T")


def retry_connect(
    fn: Callable[[], T],
    *,
    max_attempts: int = 5,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 10.0,
    label: str = "Ollama",
) -> T:
    """
    Call *fn* repeatedly until it succeeds or max_attempts is exhausted.

    Only retries on errors in RETRYABLE. Any other exception (including
    ReadTimeout and HTTP 4xx/5xx) propagates immediately.

    Parameters
    ----------
    fn:
        Zero-argument callable to retry.
    max_attempts:
        Maximum number of total attempts (including the first).
    initial_delay:
        Seconds to wait before the second attempt.
    backoff_factor:
        Multiply the delay by this factor after each failure.
    max_delay:
        Cap on delay between attempts.
    label:
        Human-readable name used in log messages.
    """
    delay = initial_delay
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except RETRYABLE as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            logger.warning(
                "%s not reachable (attempt %d/%d): %s — retrying in %.1fs",
                label, attempt, max_attempts, exc, delay,
            )
            time.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)

    raise ConnectionError(
        f"Could not connect to {label} after {max_attempts} attempts. "
        f"Is `ollama serve` running? Last error: {last_exc}"
    ) from last_exc


async def async_retry_connect(
    fn: Callable[[], "asyncio.Coroutine[None, None, T]"],
    *,
    max_attempts: int = 5,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 10.0,
    label: str = "Ollama",
) -> T:
    """Async version of retry_connect."""
    delay = initial_delay
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except RETRYABLE as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            logger.warning(
                "%s not reachable (attempt %d/%d): %s — retrying in %.1fs",
                label, attempt, max_attempts, exc, delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)

    raise ConnectionError(
        f"Could not connect to {label} after {max_attempts} attempts. "
        f"Is `ollama serve` running? Last error: {last_exc}"
    ) from last_exc
```

### Using the Retry Helper

```python
from saathi.retry import retry_connect
import httpx


def check_ollama_health(base_url: str = "http://localhost:11434") -> bool:
    def _check():
        r = httpx.get(f"{base_url}/api/tags", timeout=httpx.Timeout(connect=5.0, read=5.0))
        r.raise_for_status()
        return True

    return retry_connect(_check, max_attempts=6, label="Ollama")
```

### Connection vs Read Timeout Configuration

Configure `httpx.Timeout` explicitly to apply the retry logic only where appropriate:

```python
# For health checks and model listing: short timeouts on both connect and read
health_timeout = httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0)

# For inference: short connect timeout, NO read timeout
inference_timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=5.0)
```

Pass `inference_timeout` to `ChatOllama` via the `timeout` parameter. LangChain will use it for all
requests to `/api/chat`.

---

## 10. Remote Ollama

Running Ollama on a dedicated machine — a desktop with a powerful GPU, a home server, or a cloud
VM — and connecting to it from a laptop is a common and practical setup. saathi supports this
through the `SAATHI_BASE_URL` environment variable, which is passed directly to `ChatOllama`.

### Making Ollama Listen on All Interfaces

By default Ollama binds to `127.0.0.1` and refuses connections from other machines. To accept
remote connections, bind to `0.0.0.0`:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

> **Warning:** Binding to `0.0.0.0` with no authentication exposes your Ollama instance to anyone
> on the network. Never do this on a public network without additional protection (firewall,
> VPN, reverse proxy with auth). On a home LAN with trusted devices it is acceptable but still
> worth restricting by IP in the firewall.

### Firewall Configuration

On Linux with `ufw`:

```bash
# Allow connections from a specific machine only
sudo ufw allow from 192.168.1.50 to any port 11434

# Allow from entire local subnet
sudo ufw allow from 192.168.1.0/24 to any port 11434
```

With `firewalld`:

```bash
sudo firewall-cmd --permanent --add-rich-rule='
  rule family="ipv4"
  source address="192.168.1.0/24"
  port protocol="tcp" port="11434"
  accept'
sudo firewall-cmd --reload
```

### SSH Tunnel

The safest approach for remote access is an SSH tunnel. No firewall changes are needed; the
traffic is encrypted and authenticated by SSH:

```bash
# Forward local port 11434 to the remote machine's localhost:11434
ssh -L 11434:localhost:11434 -N user@gpu-server.local
```

Run this in a terminal (or background it with `-f`). Then set saathi's base URL to localhost:

```bash
export SAATHI_BASE_URL=http://localhost:11434
saathi
```

To make the tunnel persistent and auto-reconnect:

```bash
# Using autossh
autossh -M 0 -N -L 11434:localhost:11434 user@gpu-server.local
```

For a permanent setup add this to `~/.ssh/config`:

```text
Host gpu-server
    HostName gpu-server.local
    User yourname
    LocalForward 11434 localhost:11434
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

Then `ssh -N gpu-server` establishes the tunnel.

### Tailscale

Tailscale creates a peer-to-peer encrypted WireGuard mesh between your devices. Once installed
on both machines:

```bash
# On the GPU server
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# On the laptop (replace with your Tailscale IP)
export SAATHI_BASE_URL=http://100.64.1.2:11434
saathi
```

Tailscale traffic is encrypted end-to-end. Since both machines are on your personal tailnet, no
additional firewall rules are needed. This is the recommended approach for permanent remote setups.

### nginx Reverse Proxy

An nginx reverse proxy gives you HTTPS, optional basic authentication, and request logging:

```nginx
server {
    listen 443 ssl;
    server_name ollama.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/ollama.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ollama.yourdomain.com/privkey.pem;

    # Optional basic auth
    auth_basic "Ollama API";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://localhost:11434;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # Critical: disable buffering so streaming works
        proxy_buffering off;
        proxy_cache off;

        # Long timeouts for inference
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
```

> **Warning:** `proxy_buffering off` is mandatory. With buffering enabled, nginx holds the entire
> response in memory before forwarding it, which breaks streaming completely. The client receives
> nothing until the full generation is done.

Connect saathi with:

```bash
export SAATHI_BASE_URL=https://ollama.yourdomain.com
```

For further detail on remote Ollama setup, see the project's
`docs/ollama-remote.md` reference document.

---

## 11. /doctor — Health Checking Ollama

The `/doctor` command in saathi runs a series of checks against the local Ollama installation and
reports any problems with actionable fix instructions. It is the first thing to run when something
seems wrong.

### What the Diagnostics Check

`src/saathi/diagnostics.py` performs the following checks in order:

1. **Connectivity** — can we reach the Ollama daemon at the configured URL?
2. **API response** — does `/api/tags` return valid JSON?
3. **Model presence** — is the configured model in the list?
4. **Model name disambiguation** — if the exact name is not found, suggest similar names.

```python
"""
src/saathi/diagnostics.py
-------------------------
Health checks for the Ollama connection and model availability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

import httpx


class CheckStatus(Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    fix: str = ""


@dataclass
class DiagnosticsReport:
    base_url: str
    model: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.status == CheckStatus.OK for r in self.results)

    @property
    def errors(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == CheckStatus.ERROR]


def run_diagnostics(
    model: str = "gemma3:12b",
    base_url: str = "http://localhost:11434",
) -> DiagnosticsReport:
    """Run all Ollama health checks and return a report."""
    report = DiagnosticsReport(base_url=base_url, model=model)

    # --- Check 1: Connectivity ---
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
    except httpx.ConnectError:
        report.results.append(CheckResult(
            name="connectivity",
            status=CheckStatus.ERROR,
            message=f"Cannot connect to Ollama at {base_url}",
            fix=(
                "Start Ollama with `ollama serve`, or check that the daemon is running.\n"
                "On Linux: `systemctl status ollama`\n"
                "On macOS: check the menu-bar icon.\n"
                "If using a non-default address, set SAATHI_BASE_URL."
            ),
        ))
        return report  # No point running further checks
    except httpx.ConnectTimeout:
        report.results.append(CheckResult(
            name="connectivity",
            status=CheckStatus.ERROR,
            message=f"Connection to {base_url} timed out",
            fix=(
                "The host is reachable but not responding. "
                "Check that Ollama is running and not overloaded."
            ),
        ))
        return report

    report.results.append(CheckResult(
        name="connectivity",
        status=CheckStatus.OK,
        message=f"Ollama is reachable at {base_url}",
    ))

    # --- Check 2: Valid API response ---
    try:
        response.raise_for_status()
        data = response.json()
        models_list = data.get("models", [])
    except (httpx.HTTPStatusError, json.JSONDecodeError) as exc:
        report.results.append(CheckResult(
            name="api_response",
            status=CheckStatus.ERROR,
            message=f"Unexpected response from /api/tags: {exc}",
            fix="This may indicate an incompatible Ollama version. Try `ollama --version`.",
        ))
        return report

    report.results.append(CheckResult(
        name="api_response",
        status=CheckStatus.OK,
        message=f"API returned valid JSON with {len(models_list)} model(s)",
    ))

    # --- Check 3: Model presence ---
    available_names = [m["name"] for m in models_list]
    exact_match = model in available_names
    prefix_match = any(name.startswith(model.split(":")[0]) for name in available_names)

    if exact_match:
        report.results.append(CheckResult(
            name="model_presence",
            status=CheckStatus.OK,
            message=f"Model '{model}' is available",
        ))
    elif prefix_match:
        similar = [n for n in available_names if n.startswith(model.split(":")[0])]
        report.results.append(CheckResult(
            name="model_presence",
            status=CheckStatus.WARNING,
            message=f"Model '{model}' not found, but similar models are: {similar}",
            fix=(
                f"Either pull the exact model with `ollama pull {model}` "
                f"or set SAATHI_MODEL={similar[0]} to use an existing one."
            ),
        ))
    else:
        report.results.append(CheckResult(
            name="model_presence",
            status=CheckStatus.ERROR,
            message=f"Model '{model}' is not installed",
            fix=(
                f"Pull the model with: ollama pull {model}\n"
                f"Available models: {', '.join(available_names) or '(none)'}"
            ),
        ))

    return report


def format_report(report: DiagnosticsReport) -> str:
    """Format a DiagnosticsReport as human-readable text."""
    lines = [f"Saathi Diagnostics — {report.base_url}", "=" * 50]

    icons = {CheckStatus.OK: "✓", CheckStatus.WARNING: "⚠", CheckStatus.ERROR: "✗"}

    for result in report.results:
        icon = icons[result.status]
        lines.append(f"{icon}  {result.name}: {result.message}")
        if result.fix:
            for fix_line in result.fix.splitlines():
                lines.append(f"   → {fix_line}")

    lines.append("")
    if report.ok:
        lines.append("All checks passed. saathi is ready.")
    else:
        error_count = len(report.errors)
        lines.append(f"{error_count} error(s) found. Fix the issues above and run /doctor again.")

    return "\n".join(lines)
```

### The /doctor Command Implementation

In `src/saathi/cli.py`, the `/doctor` command is a custom command that calls `run_diagnostics` and
formats the result:

```python
from saathi.diagnostics import run_diagnostics, format_report


def handle_doctor_command(model: str, base_url: str) -> None:
    """Handle the /doctor slash command."""
    report = run_diagnostics(model=model, base_url=base_url)
    print(format_report(report))
    if not report.ok:
        raise SystemExit(1)
```

### Reading the Diagnostic Output

A healthy `/doctor` output looks like:

```text
Saathi Diagnostics — http://localhost:11434
==================================================
✓  connectivity: Ollama is reachable at http://localhost:11434
✓  api_response: API returned valid JSON with 2 model(s)
✓  model_presence: Model 'gemma3:12b' is available

All checks passed. saathi is ready.
```

A failed output with actionable guidance:

```text
Saathi Diagnostics — http://localhost:11434
==================================================
✗  connectivity: Cannot connect to Ollama at http://localhost:11434
   → Start Ollama with `ollama serve`, or check that the daemon is running.
   → On Linux: `systemctl status ollama`
   → On macOS: check the menu-bar icon.
   → If using a non-default address, set SAATHI_BASE_URL.

1 error(s) found. Fix the issues above and run /doctor again.
```

---

## 12. Monitoring Ollama

Once Ollama is running in production you want visibility into three things: how fast it is going,
how much GPU memory it is using, and whether the model is loading correctly.

### Token Throughput

Ollama includes performance statistics in the final event of each streaming response. The key fields
are:

- `eval_count` — number of tokens generated (not including the prompt).
- `eval_duration` — time in nanoseconds spent generating those tokens.
- `prompt_eval_count` — number of tokens in the prompt.
- `prompt_eval_duration` — time spent processing the prompt.
- `load_duration` — time spent loading the model (zero if it was already loaded).

```python
def extract_performance(final_event: dict) -> dict:
    """Extract human-readable performance metrics from the final streaming event."""
    eval_count = final_event.get("eval_count", 0)
    eval_ns = final_event.get("eval_duration", 1)
    prompt_count = final_event.get("prompt_eval_count", 0)
    prompt_ns = final_event.get("prompt_eval_duration", 1)
    load_ns = final_event.get("load_duration", 0)

    return {
        "tokens_generated": eval_count,
        "generation_tok_per_sec": eval_count / (eval_ns / 1e9),
        "prompt_tokens": prompt_count,
        "prompt_tok_per_sec": prompt_count / (prompt_ns / 1e9),
        "model_load_sec": load_ns / 1e9,
        "total_sec": final_event.get("total_duration", 0) / 1e9,
    }
```

A practical benchmark target for a 12B model:

- On an Apple M3 Max: 40–60 tokens/sec
- On an NVIDIA RTX 4090: 90–130 tokens/sec
- On an NVIDIA RTX 3080: 50–80 tokens/sec
- CPU-only (Ryzen 9): 5–15 tokens/sec

### VRAM Usage with nvidia-smi

Monitor GPU memory while running inferences:

```bash
# One-shot snapshot
nvidia-smi

# Continuous monitoring (refresh every 2 seconds)
nvidia-smi dmon -s mu -d 2

# Watch specific fields
watch -n 2 'nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv'
```

For scripting, use the CSV output format:

```bash
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu \
           --format=csv,noheader,nounits
```

This prints three numbers per GPU: used MB, free MB, GPU utilisation percent.

On Apple Silicon, use:

```bash
sudo powermetrics --samplers gpu_power -n 1
```

Or the Activity Monitor's GPU History panel.

### Parsing Ollama's Verbose Log Output

Set the log level to debug to see per-layer load information:

```bash
OLLAMA_DEBUG=1 ollama serve 2>&1 | tee /tmp/ollama.log
```

Key patterns to look for in the log:

```text
# Model loading success
llm_load_tensors: offloaded 0/43 layers to GPU  ← all layers on GPU (good)
llm_load_tensors: offloaded 30/43 layers to GPU ← partial offload (some on CPU)
llm_load_tensors: offloaded 0/43 layers to GPU  ← no GPU (all on CPU — check CUDA)

# CUDA initialisation
ggml_cuda_init: GGML_CUDA_FORCE_MMQ = 0
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes

# Metal initialisation (Apple Silicon)
ggml_metal_init: allocating
ggml_metal_init: found device: Apple M3 Max

# Inference stats per request
llama_perf_context_print:        load time =    234.50 ms
llama_perf_context_print:      sample time =     12.30 ms /   256 runs
llama_perf_context_print: prompt eval time =    890.00 ms /   128 tokens
llama_perf_context_print:        eval time =   4800.00 ms /   256 tokens
```

If you see `offloaded 0/N layers to GPU` on a machine that has a GPU, the CUDA or Metal back-end
failed to initialise. Check that the CUDA toolkit is installed and that the Ollama version matches
your toolkit version.

---

## 13. GGUF and Model Files

Understanding how Ollama stores models on disk helps you plan storage, find specific model files
for manual inspection, and troubleshoot download failures.

### Where Ollama Stores Models

| Platform | Model storage path |
| --- | --- |
| Linux | `~/.ollama/models/` |
| macOS | `~/.ollama/models/` |
| Windows | `%USERPROFILE%\.ollama\models\` |

The models directory has two subdirectories:

```text
~/.ollama/models/
├── blobs/
│   ├── sha256-8ab4849b038c...   ← GGUF model weights
│   ├── sha256-e3b0c44298fc...   ← text config / template
│   └── sha256-a87ff429e2...     ← system prompt / license
└── manifests/
    └── registry.ollama.ai/
        └── library/
            └── gemma3/
                └── 12b           ← manifest file (JSON)
```

### The Manifest File

The manifest is a JSON file that lists the blobs that compose a model:

```json
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.ollama.image.manifest.v2+json",
  "config": {
    "mediaType": "application/vnd.ollama.image.model",
    "digest": "sha256:e3b0c44298fc...",
    "size": 0
  },
  "layers": [
    {
      "mediaType": "application/vnd.ollama.image.model",
      "digest": "sha256:8ab4849b038c...",
      "size": 8070044160
    },
    {
      "mediaType": "application/vnd.ollama.image.template",
      "digest": "sha256:a87ff429e2fc...",
      "size": 1234
    },
    {
      "mediaType": "application/vnd.ollama.image.system",
      "digest": "sha256:cbb5a0d4d605...",
      "size": 512
    }
  ]
}
```

To find which GGUF file a model uses:

```bash
# Print the manifest
cat ~/.ollama/models/manifests/registry.ollama.ai/library/gemma3/12b

# The largest blob is the GGUF weights file; find it by digest
ls -lh ~/.ollama/models/blobs/ | sort -k5 -rh | head -5
```

### Disk Space Planning

| Model | Quantisation | Approximate Size |
| --- | --- | --- |
| Llama 4 Scout 17B | Q4_K_M | ~10 GB |
| Gemma 3 27B | Q4_K_M | ~16 GB |
| Gemma 3 12B | Q4_K_M | ~8 GB |
| Gemma 3 4B | Q4_K_M | ~3 GB |
| Qwen2.5 Coder 7B | Q4_K_M | ~5 GB |
| Phi-4 14B | Q4_K_M | ~9 GB |
| Mistral 7B v0.3 | Q4_K_M | ~5 GB |
| Llama 3.2 3B | Q4_K_M | ~2 GB |

A practical rule: allocate 1.5× the model's VRAM footprint as disk space (to accommodate the
download cache before the final file is placed in the blobs directory).

### Quantisation Formats

GGUF supports many quantisation levels. The naming convention is:

- **Q8_0** — 8-bit quantisation; best quality, highest memory usage.
- **Q6_K** — 6-bit; excellent quality, slightly smaller than Q8.
- **Q5_K_M** — 5-bit medium; good balance.
- **Q4_K_M** — 4-bit medium; the community standard for most use cases.
- **Q3_K_M** — 3-bit medium; noticeably degraded on complex tasks.
- **Q2_K** — 2-bit; significant quality loss, use only when RAM is very tight.

The `_K` variants use k-quants, which apply different quantisation levels to different weight
matrices based on their importance to model quality. `_M` is the medium version of the k-quant
strategy. `_K_M` is almost always the right default choice.

### Copying the GGUF File

If you want to use a model's GGUF file directly with llama.cpp or another tool:

```bash
# Find the main blob (largest file)
BLOB=$(ls -S ~/.ollama/models/blobs/ | head -1)
cp ~/.ollama/models/blobs/$BLOB ~/my-model.gguf
```

---

## 14. Building a Modelfile

A Modelfile is to Ollama what a Dockerfile is to Docker: a recipe that creates a reproducible,
named model configuration from an existing base. For saathi, Modelfiles let you embed the
assistant's system prompt, tune generation parameters, and share a consistent configuration across
team members without each person having to manually configure environment variables.

### Complete Modelfile Syntax Reference

````dockerfile
# FROM: required. Base model to extend. Can be a local model name,
# an Ollama registry reference, or a GGUF file path.
FROM gemma3:12b

# SYSTEM: the system prompt injected at the start of every conversation.
# Use triple-quotes for multi-line values.
SYSTEM """
Your system prompt here.
"""

# PARAMETER: set inference parameters.
# These become the defaults; they can be overridden per-request.
PARAMETER temperature    0.2     # Sampling temperature
PARAMETER top_p          0.9     # Nucleus sampling
PARAMETER top_k          40      # Top-K sampling
PARAMETER repeat_penalty 1.1     # Penalise repetition
PARAMETER num_ctx        8192    # Context window (tokens)
PARAMETER num_predict    4096    # Max output tokens (-1 = unlimited)
PARAMETER stop           "<|eot_id|>"  # Custom stop token

# TEMPLATE: the prompt template. Usually inherited from the base model.
# Only override if you need a non-standard format.
TEMPLATE """{{ if .System }}<|system|>
{{ .System }}
<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}
<|end|>
<|assistant|>
{{ end }}{{ .Response }}<|end|>
"""

# MESSAGE: few-shot examples injected before the user's message.
MESSAGE user "Show me how to open a file in Python."
MESSAGE assistant """
Here is the idiomatic Python way to open a file:

```python
with open("path/to/file.txt", "r", encoding="utf-8") as f:
    content = f.read()
```

Using `with` ensures the file is closed automatically, even if an exception occurs.
"""

````

### Complete Working Example: saathi-coder

This Modelfile creates a `saathi-coder` persona tuned for the saathi-langgraph project's use case —
a code assistant that understands Python, LangGraph, and LangChain:

````dockerfile
# saathi-coder Modelfile
# Build: ollama create saathi-coder -f ./Modelfile
# Use:   SAATHI_MODEL=saathi-coder saathi

FROM gemma3:12b

SYSTEM """
You are Saathi, an expert AI programming assistant specialised in Python, LangGraph,
LangChain, and Ollama. You are running locally on the user's machine as part of the
saathi-langgraph project.

Your core behaviours:
- Give concise, correct answers. Prefer working code over lengthy explanation.
- Always provide complete, runnable examples — not snippets that require guessing.
- Explain the *why*, not just the what. Point out potential bugs, edge cases, and
  anti-patterns when you see them.
- When reading files, always check the encoding and handle potential errors.
- For async code, always use `asyncio.run()` at the top level, never in nested functions.
- For LangGraph, prefer StateGraph with typed state annotations.
- For LangChain tools, always add a proper docstring — it becomes the tool's description.

What you do NOT do:
- Invent file paths or function signatures you have not seen in context.
- Suggest installing packages that are already in the project's pyproject.toml.
- Override user decisions about architecture without being asked.
- Use deprecated LangChain v0.1 imports — always use langchain_core, langchain_ollama,
  and langgraph from their current packages.

Project context:
- The project is a LangGraph-based AI coding assistant CLI.
- Main source tree: src/saathi/
- Entry point: src/saathi/cli.py
- Agent graph: src/saathi/agent/graph.py
- Diagnostics: src/saathi/diagnostics.py
- Retry logic: src/saathi/retry.py
- Python version: 3.11+
"""

PARAMETER temperature    0.15
PARAMETER top_p          0.85
PARAMETER top_k          40
PARAMETER repeat_penalty 1.05
PARAMETER num_ctx        16384
PARAMETER num_predict    8192

MESSAGE user "I'm getting a ConnectionError when I try to start a conversation."
MESSAGE assistant """
This is almost always Ollama not running. Run `/doctor` to check:

```text
/doctor
```

If connectivity fails, start Ollama:

- Linux: `sudo systemctl start ollama`
- macOS: open the Ollama app or run `ollama serve`
- Windows: launch Ollama from the Start menu

Then retry. If you're connecting to a remote instance, check that `SAATHI_BASE_URL`
points to the right host and that the firewall allows port 11434.
"""
````

Build and test:

```bash
# Build the model
ollama create saathi-coder -f ./Modelfile

# Verify it appears in the list
ollama list | grep saathi-coder

# Quick test
ollama run saathi-coder "What is LangGraph's StateGraph?"

# Use with saathi
SAATHI_MODEL=saathi-coder saathi
```

### Sharing Modelfiles with a Team

Commit the Modelfile to version control. Team members can build it with a single command. This
ensures everyone uses the same system prompt and parameters without manual configuration:

```bash
# First-time setup
ollama create saathi-coder -f ./Modelfile

# After pulling updates
git pull
ollama create saathi-coder -f ./Modelfile  # Recreates with new config
```

> **Note:** `ollama create` is idempotent. Running it again with an updated Modelfile replaces
> the existing model. The old blobs are not deleted immediately but are cleaned up when you run
> `ollama prune` or when Ollama's garbage collection runs.

---

## 15. Ollama vs vLLM vs llama.cpp Directly

Choosing the right inference server depends on your use case. This section compares the three main
options objectively so you can understand why saathi chose Ollama and when you might want to
switch.

### Comparison Table

| Criterion | Ollama | vLLM | llama.cpp directly |
| --- | --- | --- | --- |
| **Installation** | Single binary, one-line installer | `pip install vllm`; requires CUDA | Compile from source or download binary |
| **macOS / Apple Silicon** | First-class (Metal) | No (Linux/NVIDIA only) | Yes (Metal) |
| **Windows support** | Yes (installer) | Experimental | Yes (binary) |
| **NVIDIA GPU** | CUDA, full acceleration | CUDA, highly optimised | CUDA |
| **AMD GPU** | ROCm (Linux) | ROCm (Linux) | ROCm, Vulkan |
| **CPU-only inference** | Yes (BLAS) | Limited | Yes (best support) |
| **GGUF model format** | Native | No (uses HF safetensors) | Native |
| **HuggingFace models** | Requires conversion to GGUF | Direct (no conversion) | Requires conversion |
| **Continuous batching** | No | Yes (key advantage) | No |
| **Concurrent requests** | Queue (OLLAMA_NUM_PARALLEL) | True parallelism | No |
| **OpenAI API compatibility** | Partial (`/v1/chat/completions`) | Full | Via `llama-server` |
| **LangChain integration** | `ChatOllama` (native) | `ChatOpenAI` (via compat API) | `ChatOpenAI` (via compat API) |
| **Model management** | `ollama pull` / registry | Manual download | Manual download |
| **Multi-GPU tensor parallel** | No | Yes | Experimental |
| **Quantisation** | GGUF (4-bit to 8-bit) | GPTQ, AWQ, FP8 | GGUF |
| **Throughput (single user)** | Excellent | Very good | Excellent |
| **Throughput (many users)** | Degrades (queue) | Scales well | Degrades |
| **Production readiness** | Good for small teams | Enterprise-grade | Experiment / development |
| **Memory efficiency** | Good (quantised GGUF) | Moderate (FP16/BF16) | Best (quantised GGUF) |
| **Ease of use** | ★★★★★ | ★★★☆☆ | ★★☆☆☆ |

### When to Use Ollama

Ollama is the right choice when:

- You are building a developer tool for individual or small-team use.
- You want to run on macOS or Windows without managing CUDA installations.
- The model you want is in GGUF format (most open models are).
- You want zero-configuration model management (`ollama pull`, done).
- You need to run offline without internet access.
- You want a simple `ChatOllama` integration in LangChain without adapters.

### When to Use vLLM

vLLM is the right choice when:

- You are deploying an API server for multiple concurrent users.
- You have a dedicated Linux server with one or more NVIDIA GPUs.
- You need true continuous batching to maximise GPU utilisation.
- Your model is on HuggingFace Hub and you want to serve it without conversion.
- You need FP8 or AWQ quantisation for server-grade performance.
- Throughput per dollar is your primary metric.

### When to Use llama.cpp Directly

llama.cpp is the right choice when:

- You are doing research on model behaviour and need low-level control.
- You want to script exact prompt formats without a server layer.
- You are running on very constrained hardware and every MB of overhead matters.
- You want to experiment with custom quantisation strategies.
- You are building a custom inference pipeline and Ollama's API is not flexible enough.

### Why saathi Chose Ollama

saathi-langgraph is a developer tool for individual developers. Its design goals are:

1. **Zero-friction installation** — a user should be able to run `pip install saathi` and `ollama
   pull gemma3:12b` and be productive within five minutes, on any platform.
2. **Offline operation** — no data leaves the user's machine. This requires a local inference
   server, not a cloud API.
3. **macOS first** — many of its target users are on Apple Silicon laptops, which rules out
   vLLM.
4. **Single-user workload** — saathi is an interactive REPL, not a multi-tenant API. The queue
   model of `OLLAMA_NUM_PARALLEL=1` is perfectly adequate.

vLLM would give better throughput on multi-GPU Linux servers, but that is not the target
environment. llama.cpp directly would save the small overhead of the Ollama process, but it would
lose model management, keep-alive, and the simple HTTP API that `ChatOllama` relies on.

Ollama hits the sweet spot: production-quality serving, cross-platform, with a managed model
registry and a clean API that LangChain integrates with natively.

---

## Summary

This chapter covered Ollama from installation to production deployment:

- **Architecture**: Ollama wraps llama.cpp with an HTTP API, automatic GPU detection, and a model
  registry. It is the simplest path from a GGUF file to a working inference endpoint.

- **Installation**: one-line installer on Linux, drag-and-drop on macOS, `.exe` on Windows. The
  `OLLAMA_HOST` environment variable controls the listen address.

- **HTTP API**: three essential endpoints — `/api/chat` for message-based inference, `/api/tags`
  for model listing, `/api/pull` for downloading models. The streaming format is
  newline-delimited JSON.

- **ChatOllama**: LangChain's wrapper around the `/api/chat` endpoint. The `make_llm()` factory in
  `src/saathi/agent/graph.py` reads configuration from environment variables.

- **Streaming**: `.astream()` yields `AIMessageChunk` objects. Use `rich.live.Live` for real-time
  terminal output. Use `astream_events()` for event-typed streams in LangGraph nodes.

- **Tool calling**: supported by Gemma 3/4, Llama 4, Qwen2.5, and Mistral. The `.bind_tools()`
  method attaches JSON schemas; tool calls arrive in `response.tool_calls`.

- **Performance**: `OLLAMA_KEEP_ALIVE` eliminates cold starts. `OLLAMA_FLASH_ATTENTION=1` reduces
  memory at long context. `num_ctx` is the biggest single lever on VRAM usage.

- **Connection resilience**: only `httpx.ConnectError` and `httpx.ConnectTimeout` should be
  retried. `httpx.ReadTimeout` means the model is thinking and must never be retried.
  `src/saathi/retry.py` encodes this with the `RETRYABLE` tuple.

- **Remote access**: SSH tunnels are the safest approach. Tailscale works well for permanent
  setups. nginx with `proxy_buffering off` enables HTTPS with streaming.

- **Diagnostics**: `src/saathi/diagnostics.py` checks connectivity, API validity, and model
  presence. The `/doctor` command surfaces these checks interactively.

- **Modelfiles**: create reproducible model configurations with custom system prompts and
  parameters. Commit them to version control for team consistency.

- **Ollama vs vLLM vs llama.cpp**: Ollama is optimal for single-user developer tools on any
  platform. vLLM wins for multi-user server deployments on Linux. llama.cpp is best for
  research and low-level control.

---

## Further Reading

- **Ollama documentation**: [https://ollama.com/docs](https://ollama.com/docs)
- **Ollama GitHub repository**: [https://github.com/ollama/ollama](https://github.com/ollama/ollama)
- **llama.cpp**: [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- **LangChain ChatOllama API reference**:
  [https://python.langchain.com/docs/integrations/chat/ollama/](https://python.langchain.com/docs/integrations/chat/ollama/)
- **GGUF format specification**:
  [https://github.com/ggml-org/ggml/blob/master/docs/gguf.md](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- **vLLM documentation**: [https://docs.vllm.ai](https://docs.vllm.ai)
- **Ollama Modelfile syntax**:
  [https://github.com/ollama/ollama/blob/main/docs/modelfile.md](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
- **httpx timeout documentation**:
  [https://www.python-httpx.org/advanced/timeouts/](https://www.python-httpx.org/advanced/timeouts/)
- **saathi-langgraph remote Ollama guide**:
  `C:\development\Python\Neural Network\saathi-langgraph\docs\ollama-remote.md`
