# saathi-cli

*Saathi means companion in Hindi. This tool walks alongside you in your codebase.*

A coding agent you own completely — built from first principles using Gemma 4, Ollama and LangChain. It reads files, writes code, navigates directories, runs shell commands and searches your codebase, all from a single terminal session with a rich, coloured interface.

This is not a replacement for Claude Code or GitHub Copilot. It is a transparent foundation you can read, modify and extend in any direction.

---

## Architecture

### Component map

```mermaid
graph LR
    classDef cExt  fill:#334155,stroke:#64748b,color:#f1f5f9
    classDef cMem  fill:#6d28d9,stroke:#7c3aed,color:#f5f3ff
    classDef cPmt  fill:#1d4ed8,stroke:#2563eb,color:#eff6ff
    classDef cTool fill:#b45309,stroke:#d97706,color:#fffbeb
    classDef cAgt  fill:#065f46,stroke:#059669,color:#ecfdf5
    classDef cCli  fill:#0e7490,stroke:#0891b2,color:#ecfeff

    USER[User]:::cExt
    OLLAMA[Ollama]:::cExt

    subgraph mem [Memory Store]
        GLOBAL[global]:::cMem
        PROJECT[project]:::cMem
    end

    subgraph prompt [System Prompt]
        SP[build prompt]:::cPmt
    end

    subgraph tools [Tools]
        FTOOL[file tools]:::cTool
        MTOOL[memory tools]:::cTool
    end

    subgraph agent [Agent]
        BUILD[build agent]:::cAgt
        STREAM[stream]:::cAgt
    end

    subgraph cli [CLI]
        INPUT[input]:::cCli
        HISTORY[history]:::cCli
        SPINNER[spinner]:::cCli
        RENDERER[renderer]:::cCli
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

#### Blocks

| Block | What it is |
| --- | --- |
| **User** | You — typing tasks and reading responses in the terminal. |
| **Ollama** | The local LLM server. Runs Gemma 4 on your machine. The agent sends chat requests here and streams tokens back. |
| **Memory Store** | Two persistent JSON files. `global` lives at `~/.saathi/memory.json` and holds facts that apply across all projects (preferred style, language). `project` lives at `.saathi/memory.json` inside the current folder and holds project-specific facts (entry point, stack, key files). |
| **System Prompt** | `build_system_prompt()` assembles the agent's identity, behavioural rules, optional context-scope paths, and the current memory block into a single string that heads every conversation. |
| **Tools — file tools** | `read_file`, `write_file`, `patch_file`, `list_directory`, `run_bash`, `search_in_file`, `search_across_files`. Each is a `@tool` function the agent can call to interact with the filesystem and shell. |
| **Tools — memory tools** | `save_memory` and `recall_memory`. Let the agent read and write facts to the memory store during a session, so useful discoveries persist across restarts. |
| **Agent — build agent** | `build_agent()` wires the LLM, tools, and system prompt into a LangGraph `CompiledStateGraph`. Called once at startup and again whenever the context scope or memory changes. |
| **Agent — stream** | The running ReAct loop. Sends the message history to Ollama, receives tool calls and observations in a loop, and emits chunks back to the CLI as they arrive. |
| **CLI — input** | `Prompt.ask()` — reads the user's task from the terminal. Also handles slash commands (`/context`, `/memory`) before they reach the agent. |
| **CLI — history** | The growing list of `HumanMessage` and `AIMessage` objects for the session. `compact_history()` trims the oldest messages before each call so the conversation always fits within the context window. |
| **CLI — spinner** | `ThinkingSpinner` — a background thread that cycles through 70+ phrases while the model is generating. Updates to show the active tool name when the agent picks a tool. |
| **CLI — renderer** | Prints tool observations in dim bordered panels. Streams the final answer token-by-token as it arrives, falling back to full Markdown rendering when the model doesn't stream. |

#### Edges

| Edge | What flows |
| --- | --- |
| `global / project → build prompt` | Saved facts are formatted into a memory block and injected into the system prompt at startup. |
| `build prompt → build agent` | The assembled system prompt string is passed into `build_agent()` so the agent knows its identity and remembered facts before the first message. |
| `file tools / memory tools → build agent` | The full tool list is registered with the agent at build time. The agent discovers tool names and docstrings automatically — no extra wiring needed. |
| `memory tools → global / project` | During a session the agent can call `save_memory` and `recall_memory` to read and write the JSON files directly. |
| `build agent → stream` | `build_agent()` returns the compiled LangGraph graph. `stream()` on that graph runs the ReAct loop. |
| `stream → Ollama` | Each iteration of the loop sends the current message list to Ollama as a chat request and reads back a token stream. |
| `User → input` | The user presses Enter; the terminal captures the text. |
| `input → history` | The new `HumanMessage` is appended to the session history list. |
| `history → stream` | `compact_history()` trims the list to 75% of the context window, then the trimmed messages are sent into the stream. |
| `stream → spinner` | When the agent emits a tool call chunk, the spinner updates to show the tool name and arguments. |
| `stream → renderer` | Tool observations are printed as panels as they arrive. The final answer is rendered as Markdown once the stream ends. |

---

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

#### Steps

| Step | What happens |
| --- | --- |
| `CLI → Mem: load memory` | At startup, `MemoryStore` reads both JSON files. Global facts and project facts are merged, with project facts taking priority for the same key. |
| `Mem → CLI: memory block` | The facts are formatted as a plain-text block (`key: value` lines) ready to be injected into the system prompt. |
| `CLI → Agent: build with memory` | `build_agent()` is called with the memory block. The system prompt now contains the agent's identity, any context-scope paths, and all remembered facts. |
| `User → CLI: task` | You type a task and press Enter. Slash commands are handled here before the agent sees anything. |
| `CLI → Agent: message history` | `compact_history()` trims the session history to fit 75% of the 32k context window, always keeping the system prompt and the most recent messages. The trimmed list is passed to `agent.stream()`. |
| `Agent → Ollama: chat request` | LangGraph sends the message list to Ollama's `/api/chat` endpoint. Gemma 4 receives the system prompt, conversation history, tool definitions, and your task. |
| `Ollama → Agent: response` | Gemma streams tokens back. If the response contains a tool call, LangGraph intercepts it, executes the tool, appends the observation, and sends another request. This loop repeats until Gemma produces a plain text answer. |
| `Agent → CLI: final answer` | The last message in the stream with no tool calls is the final answer. It arrives as a text chunk. |
| `CLI → User: render response` | Rich renders the answer as full Markdown — headers, code blocks, tables, and inline formatting — under a cyan rule. |
| `CLI → Mem: save if useful` | If the agent called `save_memory` during the turn, the new fact is already written to the JSON file and will be available in every future session. |

---

## File structure

```text
saathi-cli/
├── cli.py            main loop, banner, help, agent streaming, arg parsing
├── commands.py       one handler per slash command, SessionState dataclass, ThinkingSpinner
├── agent.py          LLM connection, context window config, build_agent(), compact_history()
├── system_prompt.py  agent persona, mode presets, context-scope injection, memory injection
├── tools.py          ten tool definitions (read, write, patch, list, bash, search, search_across, search_web, save_memory, recall_memory)
├── memory_store.py   persistent memory — global (~/.saathi/) and project-level (.saathi/)
├── requirements.txt  Python dependencies
└── README.md         this file
```

| File | What it owns |
| --- | --- |
| `cli.py` | The main loop, banner, help text, agent streaming with Live Markdown, and `__main__` entry point. Thin by design — dispatches every slash command to `commands.py`. |
| `commands.py` | One handler function per slash command. `SessionState` dataclass holds all mutable session variables. `ThinkingSpinner` and display helpers live here too. |
| `agent.py` | Connects to Ollama, accepts a `model_id` parameter, sets context window size, builds the LangGraph agent, compacts history. |
| `system_prompt.py` | Agent identity, behavioural rules, `MODE_ADDENDA` for `/mode` presets, context-scope and memory injection. |
| `tools.py` | Each `@tool` function is a discrete capability. Add new tools here — the agent discovers them automatically. |
| `memory_store.py` | Reads and writes `memory.json` files at global and project scope. Independent of LangChain. |

---

## Tools available

| Tool | What it does |
| --- | --- |
| `read_file` | Read any file's full contents (up to 100 KB) |
| `write_file` | Write or overwrite a file; creates parent directories |
| `patch_file` | Apply a unified diff patch to a file without rewriting it |
| `list_directory` | List files and folders with sizes |
| `run_bash` | Execute a shell command, capture stdout + stderr |
| `search_in_file` | Find text in a file, returns matching lines with line numbers |
| `search_across_files` | Grep a pattern recursively across an entire directory |
| `search_web` | Search the web via DuckDuckGo (or Brave with an API key) |
| `save_memory` | Persist a fact to global or project memory |
| `recall_memory` | Retrieve all saved facts from memory |

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
| `/paste` | Enter multi-line input mode (blank line to send, Ctrl+C to cancel) |
| `/model <model-id>` | Switch to a different Ollama model mid-session |
| `/mode explain` | Tune agent for clear explanations — prefers reads, never modifies files |
| `/mode refactor` | Tune agent for code quality — explains every change, runs tests |
| `/mode debug` | Tune agent for root-cause debugging — reproduces before fixing |
| `/mode` | Show current mode |
| `/mode off` | Clear mode — return to default behaviour |
| `/diff` | Show a unified diff of every file changed in this session |
| `/export` | Dump conversation history to a timestamped markdown file |
| `/copy` | Copy the last agent response to the clipboard |
| `/session save <name>` | Save the current session — history, context, and checkpoints |
| `/session load <name>` | Restore a previously saved session |
| `/session list` | List all saved sessions |
| `/memory list` | Show all saved facts (global and project) |
| `/memory save <scope> <key> <value>` | Manually save a fact |
| `/memory delete <scope> <key>` | Delete a fact |
| `/memory clear <scope>` | Wipe all facts from a scope |
| `clear` | Reset conversation history (keeps scope and memory) |
| `/compact` | Summarise history with the LLM — frees tokens while preserving context |
| `/rollback` | Undo the last turn — restores any files the agent changed and removes the turn from history |
| `/rollback <n>` | Undo the last n turns |
| `/checkpoints` | List all turns and which files each one touched |
| `help` | Show command reference |
| `quit` / `exit` | End the session |

### Startup flags

```bash
python cli.py --model gemma4:27b
python cli.py --context ./src ./lib/utils.py
python cli.py --model gemma4:4b --context ./experiments
```

---

## Rollback

Every turn is checkpointed before the agent runs. If you don't like what the agent did — it rewrote a file incorrectly, went in the wrong direction, or just made a mess — you can undo it.

### How it works

Before each turn, saathi snapshots the original content of every file the agent is about to touch. After the turn completes, that snapshot is saved alongside the position in conversation history. Rolling back replays those snapshots in reverse.

- Files the agent **edited** are restored to their content before the turn.
- Files the agent **created** are deleted.
- The turn's messages are removed from conversation history so the agent has no memory of the undone work.

Shell commands (`run_bash`) are **not** reversible — installs, git commits, and other side effects happen outside saathi's control.

### Commands

```text
You: /rollback
You: /rollback 3
You: /checkpoints
```

| Command | What it does |
| --- | --- |
| `/rollback` | Undo the last turn |
| `/rollback <n>` | Undo the last n turns |
| `/checkpoints` | Show a table of every recorded turn and which files it touched |

### Example

```text
You: Refactor utils.py to use dataclasses

⚙ read_file(file_path='utils.py')
⚙ write_file(file_path='utils.py', ...)

────────────── saathi ──────────────
Done. Converted three classes to dataclasses.

You: Actually that broke something, undo it

You: /rollback
  restored  /path/to/utils.py
Rolled back 1 turn.

You: /rollback 2       ← undo two turns at once
  restored  /path/to/utils.py
  deleted   /path/to/new_helper.py
Rolled back 2 turns.
```

### Checkpoints table

`/checkpoints` shows you what is available to roll back before you commit to it:

```text
┌─────────────────────────────────────────────┐
│ Checkpoints                                 │
├──┬──────────────────────────────┬───────────┤
│ #│ Task                         │ Files     │
├──┼──────────────────────────────┼───────────┤
│ 1│ Read agent.py and explain... │ —         │
│ 2│ Refactor utils.py to use ... │ utils.py  │
│ 3│ Create a helper module       │ helper.py │
└──┴──────────────────────────────┴───────────┘
```

Turns that only read files or ran bash commands show `—` in the Files column — there is nothing to restore for those turns, but rolling back still removes them from conversation history.

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

## New features

### Compact — `/compact`

Summarises the entire conversation history using the LLM and replaces it with a single condensed message. Tokens are freed without losing context — unlike `clear` (which discards everything) or the silent auto-trim (which drops the oldest messages).

```text
You: /compact
Compacting conversation…

Compacted 14 message(s) → 1 summary message.

╭─ summary ────────────────────────────────────────────╮
│ ## Session summary                                   │
│                                                      │
│ - Read `tools.py` — 9 tools, noted `patch_file` and │
│   `search_across_files` were recently added          │
│ - Modified `cli.py`: added `/compact` command after  │
│   the `clear` handler                                │
│ - `requirements.txt` updated with `duckduckgo-search`│
│ - Pending: update README with new features           │
╰──────────────────────────────────────────────────────╯
```

The summary is shown immediately so you can verify nothing important was lost. The session continues normally — the agent uses the summary as its history from that point.

**When to use it:**

- After a long debugging session before switching to a new task
- When the spinner slows noticeably (a sign the context window is filling up)
- Before `/session save` to keep the saved file compact

**Difference from `clear`:**

| Command | What it does | Context preserved |
| --- | --- | --- |
| `clear` | Wipes history entirely | None |
| `/compact` | Summarises history with the LLM | Yes — files, decisions, errors, state |
| Auto-trim | Silently drops oldest messages | Partially — recent turns only |

---

### Streaming output

The agent's final answer streams progressively to the terminal using Rich's `Live` display — text appears as tokens arrive and is continuously re-rendered as proper Markdown. Headers, code blocks, bold, tables, and inline code all format correctly in real time rather than appearing as raw symbols. The spinner stops and the cyan rule prints the moment the first word is ready. Long code explanations and multi-section answers feel noticeably faster.

---

### Input history navigation

Inside a session, the up and down arrow keys cycle through your previous inputs, the same as a regular shell. On Unix this uses the stdlib `readline` module. On Windows it uses `pyreadline3` (installed automatically via `requirements.txt`). If neither is available the session still works — you just lose the arrow-key navigation.

---

### Multi-line input — `/paste`

When you need to give the agent a long prompt — a stack trace, a code snippet, a detailed spec — `/paste` opens a multi-line collector. Type or paste as many lines as you like, then press Enter on a blank line to send them all as one message. Press Ctrl+C to cancel.

```text
You: /paste
Enter your message (blank line to send, Ctrl+C to cancel):
Here is the error I'm seeing:

AttributeError: 'NoneType' object has no attribute 'split'
  File "parser.py", line 42, in parse_config

The value comes from os.environ.get("CONFIG_PATH").
                          ← blank line sends
```

---

### Diff viewer — `/diff`

Shows a colour-coded unified diff of every file the agent has changed since the session started. Diffs are computed from the checkpoint snapshots, so they reflect all edits across all turns — not just the last one.

```text
You: /diff

────────────── /path/to/utils.py ──────────────
  @@ -10,6 +10,8 @@
   def parse():
  -    return None
  +    result = {}
  +    return result
```

Files that were read but not changed do not appear. If nothing has changed, saathi says so.

---

### Export conversation — `/export`

Dumps the full conversation history to a Markdown file in the current directory. Useful for archiving a debugging session, sharing a code walkthrough, or keeping a record of what the agent built.

```text
You: /export
Exported conversation to: /path/to/saathi-export-20260626-143201.md
```

The file uses `## You` and `## Saathi` headers for each message, making it readable as a standalone document.

---

### Clipboard — `/copy`

Copies the last agent response to the system clipboard so you can paste it straight into an editor, a PR description, or a chat window.

```text
You: /copy
Last response copied to clipboard.
```

Requires `pyperclip` (in `requirements.txt`). Falls back gracefully in headless environments where no clipboard is available.

---

### Model switch — `/model`

Swap the active Ollama model without restarting the session. The agent is rebuilt with the new model and conversation history is reset (old context from a different model would be misleading).

```text
You: /model gemma4:27b
Switched to model: gemma4:27b — history reset.

You: /model llama3.2:3b
Switched to model: llama3.2:3b — history reset.
```

Any model served by Ollama works. Combine with `--model` at startup to set the initial model.

---

### Session save and restore — `/session`

Save the full state of a session — conversation history, context scope, and file-change checkpoints — and restore it later. Sessions are stored as JSON in `.saathi/sessions/`.

```text
You: /session save my-refactor
Session saved to: .saathi/sessions/my-refactor.json

You: /session list
┌──────────────┬─────────────────────┐
│ Name         │ Saved at            │
├──────────────┼─────────────────────┤
│ my-refactor  │ 2026-06-26 14:32:01 │
└──────────────┴─────────────────────┘

You: /session load my-refactor
Session loaded: my-refactor — 8 messages, 3 checkpoints.
```

On load, `/rollback` still works — the checkpoints are restored so you can undo any turn from the previous session.

---

### Patch file — `patch_file` tool

The agent can now apply a targeted unified diff instead of rewriting an entire file. This uses fewer tokens, is easier to review, and produces a cleaner rollback diff.

The agent uses `patch_file` when making a small, well-defined change to an existing file. It verifies context lines match before writing — if the file has drifted from what the patch expects, it returns a precise mismatch error rather than silently corrupting the file. The original is snapshotted for rollback just like `write_file`.

---

### Web search — `search_web` tool

The agent can now look up library documentation, error messages, and API changes instead of relying on potentially stale training data. It uses DuckDuckGo by default (no API key, no setup). Set `BRAVE_API_KEY` in your environment to use Brave Search for higher-quality structured results.

```text
You: What is the correct way to use trim_messages in langchain-core 0.3?

# agent calls:
search_web("langchain-core trim_messages 0.3 usage")

# returns top 5 results: title, URL, and snippet for each
```

The tool caps output at ~2000 characters to avoid flooding the context window. The agent summarises the results rather than quoting them verbatim.

**Brave upgrade (optional):**

```bash
# Windows
$env:BRAVE_API_KEY = "your-key-here"

# Unix
export BRAVE_API_KEY=your-key-here
```

Get a free key at [brave.com/search/api](https://brave.com/search/api) — the free tier covers 2,000 queries/month.

---

### Mode presets — `/mode`

Switch the agent's behaviour for a specific task type without changing the model or clearing history. The mode appends a focused instruction block to the system prompt. The base persona (saathi's identity and tool access) stays intact.

The current mode is shown in the input prompt: `You (debug):`.

```text
You: /mode explain
Mode set: explain — agent rebuilt, history kept.

You (explain): How does compact_history work?

You: /mode refactor
Mode set: refactor — agent rebuilt, history kept.

You (refactor): Clean up the error handling in tools.py

You: /mode debug
Mode set: debug — agent rebuilt, history kept.

You (debug): Why does /rollback sometimes leave files in a broken state?

You: /mode off
Mode cleared. — agent rebuilt, history kept.

You: /mode
Current mode: (none — default behaviour)
```

| Mode | What the agent prioritises |
| --- | --- |
| `explain` | Clarity. Reads only, never modifies. File + line citations on every code reference. Plain language. |
| `refactor` | Code quality. Uses `patch_file` over `write_file`. Explains each change. Runs tests and reports results. |
| `debug` | Root cause. Reproduces the bug first. Reads the error trace. Proposes the smallest possible fix. Verifies after. |

Switching mode rebuilds the agent with the new system prompt but keeps conversation history — the agent remembers what you discussed before the switch.

---

### Search across files — `search_across_files` tool

The agent can now grep a pattern across an entire directory in a single tool call rather than calling `search_in_file` once per file.

```text
# agent calls:
search_across_files(directory="./src", pattern="def parse", file_glob="*.py")

# returns:
Found 4 match(es) for 'def parse' in './src':
  parser.py:12: def parse_config(path: str) -> dict:
  parser.py:47: def parse_env(key: str) -> str:
  utils/io.py:8: def parse_json(raw: str) -> dict:
  tests/test_parser.py:5: def parse_fixture() -> dict:
```

Binary files are skipped automatically. Results are capped at 200 matches. Supply a `file_glob` like `*.py` or `*.ts` to narrow the search.

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

- [x] Streaming final answer — text appears progressively as tokens arrive
- [x] Input history navigation — up/down arrows cycle through previous inputs
- [x] Multi-line input — `/paste` mode for pasting long prompts
- [x] Diff viewer — `/diff` shows all file changes across the session
- [x] Export conversation — `/export` writes a timestamped Markdown file
- [x] Clipboard — `/copy` copies the last response
- [x] Model switch — `/model <id>` swaps the LLM mid-session
- [x] Session save/restore — `/session save|load|list`
- [x] `patch_file` tool — targeted diff-based edits instead of full rewrites
- [x] `search_across_files` tool — recursive grep across an entire directory
- [x] `search_web` tool — DuckDuckGo by default, Brave Search with `BRAVE_API_KEY`
- [x] `/mode` presets — `explain`, `refactor`, `debug` tune the system prompt without clearing history
- [ ] Git tool (status, diff, log, commit)
- [ ] Multi-file context (auto-read all files in scoped folder at startup)
- [ ] Token usage display per turn
- [ ] MCP server support (expose tools over Model Context Protocol)

---

## Part of LLMfromScratch

This project is part of [github.com/dwinsi/LLMfromScratch](https://github.com/dwinsi/LLMfromScratch) — a series building from a single neuron to a language model to a working agent, one step at a time.

---

*Built with Gemma 4 + Ollama + LangChain + Rich. Apache 2.0 licence.*
