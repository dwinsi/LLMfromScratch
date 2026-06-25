# saathi-cli

*Saathi means companion in Hindi. This tool walks alongside you in your codebase.*

A coding agent you own completely — built from first principles using Gemma 4, Ollama and LangChain. It reads files, writes code, navigates directories, runs shell commands and searches your codebase, all from a single terminal session with a rich, coloured interface.

This is not a replacement for Claude Code or GitHub Copilot. It is a transparent foundation you can read, modify and extend in any direction.

---

## Architecture

### Component map

```mermaid
graph LR
    USER[User]
    OLLAMA[Ollama]

    subgraph mem [Memory Store]
        GLOBAL[global]
        PROJECT[project]
    end

    subgraph prompt [System Prompt]
        SP[build prompt]
    end

    subgraph tools [Tools]
        FTOOL[file tools]
        MTOOL[memory tools]
    end

    subgraph agent [Agent]
        BUILD[build agent]
        STREAM[stream]
    end

    subgraph cli [CLI]
        INPUT[input]
        HISTORY[history]
        SPINNER[spinner]
        RENDERER[renderer]
    end

    GLOBAL --> SP
    PROJECT --> SP
    SP --> BUILD
    FTOOL --> BUILD
    MTOOL --> BUILD
    MTOOL --> GLOBAL
    MTOOL --> PROJECT
    BUILD --> STREAM
    STREAM --> OLLAMA

    USER --> INPUT
    INPUT --> HISTORY
    HISTORY --> STREAM
    STREAM --> SPINNER
    STREAM --> RENDERER
```

### Request flow — one turn

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Mem
    participant Agent
    participant Ollama

    CLI->>Mem: load memory
    Mem-->>CLI: memory block
    CLI->>Agent: build with memory
    User->>CLI: task
    CLI->>Agent: message history
    Agent->>Ollama: chat request
    Ollama-->>Agent: response
    Agent-->>CLI: final answer
    CLI->>User: render response
    CLI->>Mem: save if useful
```

---

## File structure

```text
saathi-cli/
├── cli.py            terminal UI, spinner, rich rendering, history, /context command
├── agent.py          LLM connection, context window config, build_agent(), compact_history()
├── system_prompt.py  agent persona, context-scope injection, memory injection
├── tools.py          seven tool definitions (read, write, list, bash, search, save_memory, recall_memory)
├── memory_store.py   persistent memory — global (~/.saathi/) and project-level (.saathi/)
├── requirements.txt  Python dependencies
└── README.md         this file
```

| File | What it owns |
| --- | --- |
| `cli.py` | Everything the user sees and types. Spinner, colour, `/context` flag, session history. |
| `agent.py` | Connects to Ollama, sets context window size, builds the LangGraph agent, compacts history. |
| `system_prompt.py` | Agent identity, behavioural rules, context-scope injection, memory injection. |
| `tools.py` | Each `@tool` function is a discrete capability. Add new tools here — the agent discovers them automatically. |
| `memory_store.py` | Reads and writes `memory.json` files at global and project scope. Independent of LangChain. |

---

## Tools available

| Tool | What it does |
| --- | --- |
| `read_file` | Read any file's full contents (up to 100 KB) |
| `write_file` | Write or overwrite a file; creates parent directories |
| `list_directory` | List files and folders with sizes |
| `run_bash` | Execute a shell command, capture stdout + stderr |
| `search_in_file` | Find text in a file, returns matching lines with line numbers |

---

## Setup

### Step 1 — Install Ollama

Download from [ollama.com/download](https://ollama.com/download) and start the server:

```bash
ollama serve
```

### Step 2 — Pull Gemma 4

```bash
ollama pull gemma4:12b
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run

```bash
python cli.py
```

---

## Usage

### Basic session

```bash
python cli.py
```

```text
╭─────────────────────────────────────────╮
│ saathi — your coding companion          │
│ Powered by Gemma 4 via Ollama           │
╰─────────────────────────────────────────╯

You: List all Python files in this directory
You: Read agent.py and explain the streaming loop
You: Create a file called hello.py that prints Hello World and run it
You: Search for the word 'tool' in tools.py
You: What does this codebase do overall?
```

### Scope the agent to specific files or folders

```bash
# at startup — agent focuses on these paths from the beginning
python cli.py --context ./src ./utils/config.py

# mid-session — update scope without restarting
You: /context ./src/models ./tests

# clear scope — agent works unrestricted again
You: /context
```

When a context scope is set, the system prompt tells the agent to prefer those paths when the user says "this file" or "here" without naming a specific path. Changing scope also clears conversation history since old context would be misleading.

### Session commands

| Command | Description |
| --- | --- |
| `<any text>` | Run a task |
| `/context <path> ...` | Scope agent to specific files or folders |
| `/context` | Clear scope — agent works unrestricted |
| `/memory list` | Show all saved facts |
| `/memory save <scope> <key> <value>` | Manually save a fact |
| `/memory delete <scope> <key>` | Delete a fact |
| `/memory clear <scope>` | Wipe all facts from a scope |
| `clear` | Reset conversation history (keeps scope and memory) |
| `help` | Show command reference |
| `quit` / `exit` | End the session |

### Startup flags

```bash
python cli.py --model gemma4:27b
python cli.py --context ./src ./lib/utils.py
python cli.py --model gemma4:4b --context ./experiments
```

---

## Context window and token management

Ollama defaults to a 2048-token context window, which causes responses to be cut off mid-sentence. saathi-cli sets sensible defaults in `agent.py`:

```python
CTX_WINDOW  = 32768   # total tokens: prompt + history + response
MAX_PREDICT = 4096    # max tokens the model can generate per response
```

Gemma 4 supports up to 128k. Raise these values if you have the RAM — edit `agent.py` directly.

### Conversation history compaction

Each turn accumulates messages in a `history` list. Before every call, `compact_history()` trims the oldest messages to stay within 75% of `CTX_WINDOW`, always keeping:

- the system prompt
- the most recent messages
- the history starting on a `HumanMessage` (never mid-conversation)

This means the agent remembers earlier turns in the session, and long sessions degrade gracefully rather than failing silently.

---

## Persistent memory

saathi remembers facts across sessions using two JSON files:

| Scope | Location | Use for |
| --- | --- | --- |
| Global | `~/.saathi/memory.json` | User preferences, coding style, cross-project conventions |
| Project | `.saathi/memory.json` | Stack, entry points, architecture decisions, key files |

Both are loaded at startup and injected into the system prompt so the agent already knows saved facts without spending a tool call on them. Project facts override global facts for the same key.

### Managing memory during a session

Four commands let you inspect and modify memory without restarting:

**List all facts:**

```text
You: /memory list
```

Shows global and project memory in a table. Project facts override global facts for the same key.

**Save a fact manually:**

```text
You: /memory save project entry_point cli.py
You: /memory save global preferred_language Python
```

The agent calls this automatically when it learns something, but you can also save facts directly.

**Delete a single fact:**

```text
You: /memory delete project entry_point
```

**Clear an entire scope:**

```text
You: /memory clear project
```

Wipes all facts from global or project memory. Scope must be `'global'` or `'project'`.

### Memory files

```jsonc
// ~/.saathi/memory.json  (global — persists across all projects)
{
  "preferred_style": "concise answers, no preamble",
  "preferred_language": "Python"
}

// .saathi/memory.json  (project — specific to this folder)
{
  "entry_point": "cli.py",
  "llm_framework": "langchain 1.x with create_agent",
  "context_window": "32768 tokens"
}
```

You can edit these files directly with any text editor — they're plain JSON. Delete a key to forget it.

### Memory scoping

When you run `/context ./some-folder`, the project memory pointer moves to that folder's `.saathi/memory.json`. Each folder keeps its own isolated memory. Global memory is shared across all projects.

Using `/context` also clears the conversation history (but keeps memory), since old context would be misleading in a new folder.

---

## How the terminal UI works

The CLI uses [Rich](https://github.com/Textualize/rich) for all output.

- **Spinner** — cycles through 70+ phrases (`simmering…`, `philosophising…`, `jugaad laga raha hoon…`, `ulajhan suljha raha hoon…`) while the model generates
- **Tool call** — when the agent picks a tool, the spinner updates to show `⚙ read_file(file_path='agent.py')`
- **Observation panel** — tool result shown in a dim bordered panel with syntax highlighting (truncated at 300 chars)
- **Final answer** — rendered as full Markdown (headers, code blocks, tables) under a cyan rule

---

## Extending saathi

### Add a new tool

Open `tools.py` and add a `@tool` function. The docstring is what the agent reads to decide when to use it — make it precise:

```python
@tool
def fetch_url(url: str) -> str:
    """
    Fetch the content of a web page and return it as plain text.
    Use this when the user asks about something that requires live web information.
    Input: a full URL including https://.
    Returns the page text, or an error message if the request fails.
    """
    import httpx
    response = httpx.get(url, timeout=10, follow_redirects=True)
    return response.text[:5000]
```

No other changes needed — `get_all_tools()` returns all `@tool` functions automatically.

### Change the agent's personality

Edit `SYSTEM_PROMPT_BASE` in `system_prompt.py`. The prompt controls:

- how the agent introduces itself
- what rules it follows (read before write, never delete unless asked)
- how it handles uncertainty
- whether it defers to official documentation over memorised API details

### Swap the model

Any model served by Ollama works. Change `OLLAMA_MODEL` in `agent.py` or pass `--model` at startup. Models with strong instruction-following and tool-use capability work best — Mistral, Llama 3, and Qwen 2.5 Coder are good alternatives.

### Replace Ollama with a cloud model

Swap `ChatOllama` for any LangChain chat model:

```python
# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# Anthropic
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.1)
```

The rest of the code — tools, prompt, history, streaming, CLI — is unchanged.

---

## What comes next

- [ ] Persistent memory across sessions (summarise and reload on startup)
- [ ] Git tool (status, diff, log, commit)
- [ ] Web search tool
- [ ] Multi-file context (auto-read all files in scoped folder at startup)
- [ ] Token usage display per turn
- [ ] Save session transcript to file
- [ ] MCP server support (expose tools over Model Context Protocol)

---

## Part of LLMfromScratch

This project is part of [github.com/dwinsi/LLMfromScratch](https://github.com/dwinsi/LLMfromScratch) — a series building from a single neuron to a language model to a working agent, one step at a time.

---

*Built with Gemma 4 + Ollama + LangChain + Rich. Apache 2.0 licence.*
