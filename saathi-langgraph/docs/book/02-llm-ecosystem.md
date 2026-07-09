# Chapter 2 — The LLM Ecosystem: Tokens, Models, and Inference

This chapter builds the mental model you need to work confidently with large language models as infrastructure.
Saathi is, at its heart, a thin orchestration layer around an LLM running on your machine.
To understand why saathi makes the configuration choices it does — why `temperature` defaults to `0.1`,
why the context window is `32768`, why `review.py` avoids `format="json"` — you need to understand
how the underlying models work.

We will not derive the mathematics of transformers. Instead, we will build accurate intuitions:
what a token is, what a context window really means, how inference parameters shape model behaviour,
and how all these pieces interact inside the ReAct loop that drives saathi's agent. Along the way,
every claim is grounded in the actual saathi source code.

---

## What Is a Large Language Model?

A large language model is a neural network trained to predict the next token in a sequence of text.
That description sounds modest. In practice, a sufficiently large network trained on a sufficiently large
corpus develops representations of language, reasoning, code, and world knowledge that emerge from
the statistical structure of human writing. The word "large" in LLM refers to two things simultaneously:
the number of parameters in the network (hundreds of millions to hundreds of billions) and the volume
of training data (trillions of tokens).

### The Transformer Architecture

Every modern LLM is built on the Transformer architecture, introduced by Vaswani et al. in 2017.
A Transformer processes a sequence of tokens in parallel — not one word at a time as earlier recurrent
networks did. The fundamental operation is **self-attention**: for every token in the sequence, the
model computes how much every other token should influence the representation of that token. Tokens
that are grammatically or semantically related attend strongly to each other. The attention computation
produces a weighted blend of the other tokens' value vectors.

A Transformer is composed of many identical layers stacked on top of one another. Each layer has two
sublayers:

1. **Multi-head self-attention** — the attention mechanism described above, run in parallel across
   multiple independent "heads" so the model can attend to different aspects of the input simultaneously.
2. **Feed-forward network** — a small fully-connected network applied identically to each position,
   which transforms the attended representation into the next layer's input.

The number of layers is one of the primary dials that determines model capacity. A 7B parameter model
might have 32 layers with a hidden dimension of 4096. A 70B model might have 80 layers and a hidden
dimension of 8192.

### Parameters and Weights

The "parameters" of a model are all the numerical weights stored in the network: the attention
projection matrices, the feed-forward network weights, the embedding tables that map token IDs to
vectors, and the unembedding matrix that maps the final hidden state back to logits over the vocabulary.
When we say a model has 12 billion parameters, we mean there are 12 billion floating-point numbers
stored as the model's learned knowledge. These numbers were adjusted during training to minimize
prediction error on the training corpus.

At inference time (when you run the model), those weights are fixed. The only computation happening
is the forward pass: token IDs go in, logit scores for the next token come out. The model samples from
those logits (or takes the argmax) to produce the next token, appends it to the sequence, and repeats.
This autoregressive generation is why generation is sequential and why generating a long response
takes longer than generating a short one — every token requires a full forward pass.

### Quantization

By default, neural network weights are stored as 32-bit floating-point numbers (fp32). For a 12B
parameter model, that is 12 × 10⁹ × 4 bytes = 48 GB of memory. Running a 12B model unquantized
requires a GPU with at least 48 GB of VRAM — expensive hardware.

**Quantization** reduces the bit-width of the stored weights, trading a small amount of model quality
for a dramatic reduction in memory. The most common quantization schemes used by Ollama (via llama.cpp)
are:

| Format | Bits per weight | Approx. size (12B) | Quality loss |
| ---------- | ----------------- | --------------------- | -------------- |
| fp16 | 16 | 24 GB | Negligible |
| Q8_0 | 8 | 12 GB | Very small |
| Q6_K | ~6.5 | 9.8 GB | Small |
| Q5_K_M | ~5.5 | 8.2 GB | Small |
| Q4_K_M | ~4.5 | 6.7 GB | Moderate |
| Q3_K_M | ~3.5 | 5.2 GB | Noticeable |
| Q2_K | ~2.5 | 3.7 GB | Large |

The `K` in names like `Q4_K_M` refers to k-quants, a group-quantization scheme where weights are
quantized in groups of 32 or 64, with a per-group scaling factor that reduces the accuracy loss
compared to naive quantization. The `M` suffix indicates a "medium" quality variant that uses
slightly higher precision for the most sensitive layers (attention norms and embedding tables).

`Q4_K_M` is the practical sweet spot for most users: a 12B model fits comfortably in 8 GB of VRAM
with quality that is, for coding tasks, nearly indistinguishable from the fp16 reference.
Saathi defaults to `gemma4:12b`, which Ollama ships in Q4 by default.

> **Note:** VRAM is not the only memory consideration. llama.cpp also allocates a KV-cache proportional
> to the context window size. For a 12B model with a 32k context window, the KV-cache can add another
> 2–4 GB, so the practical VRAM floor for saathi's default configuration is around 10 GB.

### What the Model Actually Knows

The weights encode a compressed, statistical summary of everything in the training corpus. The model
has no access to the internet at inference time, no memory of previous conversations (unless you
provide them in the context), and no ability to execute code unless you give it a tool. All of the
intelligence is frozen in the weights at training time.

This is why the system prompt and the conversation history passed to the model matter so much:
they are the only source of context the model has. Saathi's architecture, particularly its compaction
module (`src/saathi/compaction.py`), is designed around this constraint.

---

## Tokenization

Before a language model can process text, the text must be converted to a sequence of integers.
This conversion is performed by a **tokenizer**, and the unit of conversion is a **token**.

### What Is a Token?

A token is a chunk of text — typically a common word, a word fragment, or a punctuation character.
Tokenizers are trained on large text corpora and learn a vocabulary of 32,000 to 128,000 tokens
that efficiently covers the distribution of text in the training data. Common English words are
usually single tokens. Rare words, technical identifiers, and non-English text are split into
multiple subword tokens.

Concrete examples with GPT-style tokenization:

- `"hello world"` → `["hello", " world"]` → 2 tokens
- `"function"` → `["function"]` → 1 token
- `"tokenization"` → `["token", "ization"]` → 2 tokens
- `"Ashwini"` → `["Ash", "w", "ini"]` → 3 tokens (unfamiliar name, split into subwords)
- `"\n\n"` → `["\n\n"]` → 1 token (common in code)
- A Python `def` statement of 40 characters → roughly 12–15 tokens

The practical rule of thumb that holds remarkably well across English prose and code:

```text
1 token ≈ 4 characters
```

For a 32,768-token context window, that corresponds to roughly 130,000 characters, or about 100 pages
of text. For code, which is denser in tokens per character than prose, the real-world capacity is
closer to 80 pages.

### Tokenizer Algorithms

Two tokenizer algorithms dominate:

**Byte-Pair Encoding (BPE)** — used by Llama, GPT models, and many others. BPE starts with a
byte-level vocabulary and iteratively merges the most frequent adjacent pairs into new tokens until
the desired vocabulary size is reached.

**SentencePiece** — used by Gemma and T5. SentencePiece is an unsupervised tokenizer that operates
on raw Unicode text, treating spaces as part of tokens (a token starting a word begins with `▁`).
It supports both BPE and unigram language model training.

For saathi, the specific tokenizer is determined by the model loaded in Ollama. You don't call
the tokenizer directly in your code — Ollama handles the conversion internally before sending
the prompt to the model. However, you can inspect token counts using the `tiktoken` library,
which is compatible with many BPE tokenizers:

```python
# pip install tiktoken
import tiktoken

# cl100k_base is the tokenizer used by GPT-3.5/4 — a good approximation
# for estimating counts in other BPE models like Llama.
enc = tiktoken.get_encoding("cl100k_base")

text = "def build_graph(tools: list[BaseTool], memory_store: MemoryStore) -> None:"
tokens = enc.encode(text)
print(f"Text: {repr(text)}")
print(f"Token count: {len(tokens)}")
print(f"Tokens: {[enc.decode([t]) for t in tokens]}")
```

Running that snippet produces output similar to:

```text
Text: 'def build_graph(tools: list[BaseTool], memory_store: MemoryStore) -> None:'
Token count: 19
Tokens: ['def', ' build', '_graph', '(', 'tools', ':', ' list', '[', 'Base', 'Tool', '],',
         ' memory', '_store', ':', ' Memory', 'Store', ')', ' ->', ' None', ':']
```

This illustrates several important points: Python keywords (`def`) are single tokens; common
identifiers (`tools`, `list`) are single tokens; CamelCase names (`BaseTool`, `MemoryStore`) are
split at word boundaries; and underscored names (`build_graph`, `memory_store`) are split at the
underscore.

### Why Token Count Matters

Token counts drive two costs simultaneously:

1. **Pricing** — cloud LLM providers charge per input token and per output token. A conversation
   that uses 10,000 input tokens per turn becomes expensive at scale.

2. **Context window saturation** — every model has a maximum sequence length it can process.
   If the combined length of the system prompt, conversation history, tool outputs, and new user
   message exceeds the context window, the model will either truncate older content or refuse
   to process the request.

Saathi uses a simple character-based estimator in `src/saathi/compaction.py`:

```python
_CHARS_PER_TOKEN = 4

def estimate_tokens(messages: list[BaseMessage]) -> int:
    """Rough token estimate (~4 chars per token) across message contents."""
    return sum(len(_text(m)) for m in messages) // _CHARS_PER_TOKEN
```

This is deliberately approximate. A precise count would require running the tokenizer on every
message at each compaction check, which adds latency. The 4 chars/token heuristic is conservative
enough that compaction triggers slightly before the hard limit, providing a safety margin.

> **Warning:** The 4 chars/token rule breaks down for non-English text, heavily-minified code,
> and content with many special characters. If you are using saathi on a codebase with significant
> amounts of minified JavaScript or non-Latin text, consider lowering `SAATHI_CONTEXT_WINDOW`
> or increasing the compaction frequency.

---

## Context Windows

The context window is the maximum number of tokens an LLM can attend to at once. It defines the
"working memory" of the model during a single inference call. Everything the model knows about
your current conversation — the system prompt, all previous messages, tool call results, the new
user turn — must fit within this window.

### The Evolution of Context Window Size

Context window sizes have grown dramatically over the past five years:

| Year  | Representative model         | Context window |
| ----  | --------------------         | -------------- |
| 2019  | GPT-2                        | 1,024 tokens   |
| 2020  | GPT-3                        | 2,048 tokens   |
| 2022  | GPT-3.5 (early)              | 4,096 tokens   |
| 2023  | GPT-3.5-turbo                | 16,384 tokens  |
| 2023  | GPT-4 (8k variant)           | 8,192 tokens   |
| 2023  | Claude 2                     | 100,000 tokens |
| 2024  | Gemini 1.5 Pro               | 1,000,000 tokens |
| 2024  | Llama 3.1                    | 128,000 tokens |
| 2025  | Gemma 4 27B                  | 128,000 tokens |
| 2026  | Gemma 4 12B (saathi default) | 128,000 tokens |

The physical context window of the underlying model and the context window configured for inference
are not the same thing. Ollama's `num_ctx` parameter controls how much of the model's maximum
capacity you actually allocate. Allocating more context requires more VRAM (for the KV-cache)
and makes each forward pass slightly slower.

### Why Saathi Defaults to 32,768

The setting in `src/saathi/config.py`:

```python
context_window: int = 32768
```

This corresponds to 32k tokens — 32,768 being a power of two, convenient for KV-cache alignment.
The choice balances three competing concerns:

1. **Capacity** — 32k tokens is enough to hold a long conversation, several file reads, and tool
   outputs without triggering compaction on every turn.
2. **VRAM** — with `gemma4:12b` in Q4, the KV-cache at 32k adds roughly 2–3 GB on top of the
   model weights (~6.7 GB), keeping the total below 10 GB for a typical gaming GPU.
3. **Quality** — as explained in the final section of this chapter, quality degrades as the context
   fills. 32k is large enough to be useful but not so large that the model is routinely operating
   with a half-full context it cannot fully utilise.

You can raise or lower this value via the environment variable `SAATHI_CONTEXT_WINDOW`:

```bash
# Use 64k context (requires more VRAM)
export SAATHI_CONTEXT_WINDOW=65536

# Use a tighter 16k context to save VRAM on small GPUs
export SAATHI_CONTEXT_WINDOW=16384
```

### The History Token Budget

Saathi reserves 75% of the context window for conversation history. The remaining 25% is left for
the current turn's input and the model's output:

```python
@property
def history_token_budget(self) -> int:
    return int(self.context_window * 0.75)
```

With `context_window=32768`, the history budget is 24,576 tokens. When the estimated token count
of the stored messages exceeds this budget, the compaction module summarises the older portion of
the conversation into a single `SystemMessage`.

### Context Window Capacity in Practice

A useful rule of thumb for planning:

| Context window | Approx. characters | Approx. lines of Python | Approx. pages of prose |
| -------------- | ------------------ | ----------------------- | ---------------------- |
| 4,096          | 16 KB             | ~700                   | 12                     |
| 8,192          | 32 KB             | ~1,400                 | 25                     |
| 16,384         | 65 KB             | ~2,800                 | 50                     |
| 32,768         | 131 KB            | ~5,600                 | 100                    |
| 65,536         | 262 KB            | ~11,200                | 200                    |
| 131,072        | 524 KB            | ~22,400                | 400                    |

These figures assume average Python code density (roughly 60 characters per line) and standard
English prose (roughly 1,300 characters per page). A codebase function that reads several large
files into the context will consume the budget quickly.

> **Note:** Even when the context window is not full, there is a subtler quality problem:
> the "lost in the middle" phenomenon. Models recall information at the beginning and end of the
> context much more reliably than information in the middle. This is covered in depth in the final
> section of this chapter.

---

## Inference Parameters

When you call an LLM, the generated output is not fully deterministic. The model produces a
probability distribution over the next token at each step, and a sampling strategy selects which
token to emit. The inference parameters control that sampling process.

### Temperature

Temperature is the most important single parameter for controlling model behaviour. Conceptually,
it controls the "sharpness" of the probability distribution:

- **Low temperature (0.0–0.3)**: The distribution becomes sharply peaked at the most probable
  token. Outputs are deterministic, consistent, and predictable. The model will almost always
  choose the same token in the same context. This is desirable for coding tasks, where you want
  the most likely correct answer, not creative variation.

- **High temperature (0.7–1.5)**: The distribution flattens. The model frequently chooses tokens
  that are plausible but not the single most probable, producing more varied, creative, and
  unpredictable outputs.

Saathi defaults to `temperature=0.1`, intentionally close to zero but not exactly zero.
A temperature of exactly 0.0 produces fully greedy (argmax) decoding, which can cause the model
to get "stuck" in repetitive loops. A small non-zero temperature provides the minimal randomness
that prevents degenerate outputs without meaningfully affecting the consistency of the response.

```python
# From src/saathi/config.py
temperature: float = 0.1

# From src/saathi/agent/graph.py
def make_llm(model_id: str) -> ChatOllama:
    return ChatOllama(
        model=model_id,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,   # 0.1
        num_ctx=settings.context_window,    # 32768
        num_predict=settings.max_tokens,    # 4096
    )
```

To override temperature for a more exploratory session:

```bash
export SAATHI_TEMPERATURE=0.7
saathi
```

### Top-p (Nucleus Sampling)

Top-p (also called nucleus sampling) is a complementary sampling strategy. Instead of sampling
from the full vocabulary, it samples from the smallest set of tokens whose cumulative probability
exceeds a threshold `p`. At `top_p=0.9`, the model considers only the most probable tokens that
together account for 90% of the probability mass.

Top-p prevents the model from selecting very low-probability tokens (which temperature alone does
not fully prevent). It is less commonly tuned than temperature. The Ollama default of `top_p=0.9`
is appropriate for coding tasks and saathi does not override it.

### Top-k

Top-k restricts sampling to the `k` most probable tokens, regardless of their cumulative
probability. At `top_k=40`, only the 40 most likely next tokens are considered. Top-k is a blunt
instrument compared to top-p and is less commonly used in production, but Ollama supports it via
the `top_k` parameter.

### num_predict (Max New Tokens)

`num_predict` sets the maximum number of new tokens the model will generate before stopping.
Saathi sets this to 4,096:

```python
max_tokens: int = 4096
```

This means a single model response can be at most roughly 16,000 characters — long enough for
a complete file rewrite or an extensive explanation, but bounded to prevent runaway generation.

> **Warning:** `num_predict` is a hard ceiling on output length. If the model's response is cut
> off mid-sentence, the most likely cause is hitting this limit. You can raise it with
> `SAATHI_MAX_TOKENS=8192`, but large outputs also consume more context window on subsequent turns.

### num_ctx (Context Window at Inference)

`num_ctx` is the Ollama-specific parameter that tells llama.cpp how large a KV-cache to allocate
for this inference session. It maps directly to `SAATHI_CONTEXT_WINDOW`. Setting it larger than
the model's training context does not help — the model was not trained to attend over that distance.

### Putting It All Together

The full inference configuration for saathi, expressed as a `.env` file:

```bash
# .env — saathi inference configuration
SAATHI_OLLAMA_MODEL=gemma4:12b
SAATHI_OLLAMA_BASE_URL=http://localhost:11434
SAATHI_TEMPERATURE=0.1
SAATHI_CONTEXT_WINDOW=32768
SAATHI_MAX_TOKENS=4096
SAATHI_MAX_PARALLEL_TOOLS=8
```

These values are loaded by pydantic-settings into `src/saathi/config.py` at startup.
Any environment variable overrides the default, making it easy to experiment without touching code.

---

## Chat Message Formats

Modern LLMs trained for instruction-following expect input in a structured **chat format**:
a sequence of turns, each with a role and content. The role tells the model who is speaking.
The content is the actual text.

### Message Roles

**System** — a privileged instruction that appears at the beginning of the conversation.
The model treats it as authoritative context it should follow throughout. Saathi's system
message is assembled by `src/saathi/agent/prompts.py`:

```python
BASE_PROMPT = """\
You are Saathi (साथी) — a coding companion that walks alongside you, not in front.

Your workflow:
1. Think — understand the task and what information you need
2. Use a tool — read a file, run a command, search the codebase
3. Observe — study the tool's output carefully
4. Repeat — keep using tools until you have enough context
5. Answer — give a clear, grounded response
...
"""
```

**User** — the human's turn. In saathi's REPL, each prompt you type becomes a `HumanMessage`.

**Assistant** — the model's turn. Each response the model generates is an `AIMessage`.

**Tool** — a special role for tool results. After the model emits a tool call, the tool runs,
and its output is injected back into the conversation as a `ToolMessage` with a `tool_call_id`
linking it to the specific call.

### OpenAI-Compatible JSON Format

Ollama exposes an OpenAI-compatible API. The wire format for a chat request looks like this:

```json
{
  "model": "gemma4:12b",
  "messages": [
    {
      "role": "system",
      "content": "You are Saathi — a coding companion..."
    },
    {
      "role": "user",
      "content": "Read src/saathi/config.py and explain the settings."
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "read_file",
            "arguments": "{\"path\": \"src/saathi/config.py\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_abc123",
      "content": "from pydantic_settings import BaseSettings...\n..."
    },
    {
      "role": "assistant",
      "content": "The config.py module defines a `Settings` class using pydantic-settings..."
    }
  ]
}
```

This JSON is what LangChain's `ChatOllama` serialises your `BaseMessage` list into when it makes
the HTTP request to `http://localhost:11434/v1/chat/completions`.

### Message Ordering Rules

The ordering of messages in the chat history is not arbitrary. Several invariants must be maintained:

1. The system message, if present, comes first.
2. A `ToolMessage` must immediately follow an `AIMessage` that contains a `tool_calls` field.
3. The conversation must begin and end with the correct roles — you cannot have two consecutive
   `HumanMessage` objects, nor can the sequence start with a `ToolMessage`.

Saathi's compaction logic in `src/saathi/compaction.py` is careful to cut at a user-turn boundary
for precisely this reason: if it cut in the middle of a tool call / tool result pair, the remaining
sequence would start with an orphaned `ToolMessage` whose paired `AIMessage` was summarised away,
causing a validation error.

```python
def split_for_compaction(
    messages: list[BaseMessage], keep_turns: int
) -> tuple[list[BaseMessage], list[BaseMessage]] | None:
    human_idxs = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_idxs) <= keep_turns:
        return None
    cut = human_idxs[-keep_turns]
    return messages[:cut], messages[cut:]
```

The cut index `human_idxs[-keep_turns]` points to the start of the `keep_turns`-th-from-last
user message, ensuring the retained tail always begins cleanly.

---

## Tool Calling / Function Calling

Tool calling (also called function calling) is the mechanism by which an LLM can request that
an external function be executed and its result returned. It is the foundation of agent behaviour.

### How the Model Emits a Tool Call

When a model is given a list of tools (as JSON schemas describing their names, descriptions, and
parameters), it can choose to respond with a structured tool call instead of plain text. In the
OpenAI-compatible API, this appears in the `tool_calls` field of the assistant message:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_f7g8h9",
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": "{\"path\": \"src/saathi/review.py\", \"start_line\": 1}"
      }
    }
  ]
}
```

The model has encoded its intent — "call `read_file` with these arguments" — in a structured
format that the orchestration layer can parse and execute.

### LangChain Representation

LangChain wraps these messages in typed Python objects. An `AIMessage` with tool calls looks like:

```python
from langchain_core.messages import AIMessage, ToolCall

msg = AIMessage(
    content="",  # empty when tool calls are present
    tool_calls=[
        ToolCall(
            id="call_f7g8h9",
            name="read_file",
            args={"path": "src/saathi/review.py", "start_line": 1},
        )
    ],
)
```

After the tool executes, its output is injected back as a `ToolMessage`:

```python
from langchain_core.messages import ToolMessage

tool_result = ToolMessage(
    tool_call_id="call_f7g8h9",
    content="# Multi-reviewer code review over the working git diff...\n...",
)
```

The `tool_call_id` links the result to the specific call, which matters when the model makes
multiple tool calls in a single turn (parallel tool execution).

### Parallel Tool Execution

Saathi supports parallel tool execution, configured by `max_parallel_tools: int = 8`.
When the model emits multiple tool calls in a single `AIMessage`, they are all dispatched
concurrently (up to the configured parallelism limit) and the results collected into individual
`ToolMessage` objects, one per call.

The tool node in `src/saathi/agent/tool_node.py` handles this dispatch. Each `ToolMessage`
is appended to the state in the order the calls were made, preserving the ID linkage.

### Binding Tools to the Model

In `src/saathi/agent/graph.py`, tools are bound to the model before the graph is compiled:

```python
llm = make_llm(model_id).bind_tools(tools)
```

`bind_tools` serialises each tool's name, description, and parameter schema into the JSON
format that Ollama expects, and includes them in every API request. The model has visibility
into all tools for every turn of the conversation.

---

## The ReAct Pattern

ReAct (Reason, Act, Observe) is the fundamental algorithmic pattern that makes LLM-based agents
work. It was formalised in the paper "ReAct: Synergizing Reasoning and Acting in Language Models"
(Yao et al., 2023) but the pattern predates the paper.

### The Loop

The ReAct loop proceeds as follows:

1. **Reason** — the model receives the current state (system prompt, conversation history, user
   request) and reasons about what it needs to know and what action it should take.
2. **Act** — the model emits a tool call, specifying which tool to invoke and with what arguments.
3. **Observe** — the tool executes. Its output is appended to the conversation as a `ToolMessage`.
4. **Repeat** — the model now has the observation in its context and reasons again. It can call
   more tools, call the same tool with different arguments, or decide it has enough information
   to produce a final answer.

The loop terminates when the model produces a response without any tool calls — a plain `AIMessage`
with `tool_calls=[]`.

### Tracing Through a Real Example

Suppose the user asks saathi: "Read `src/saathi/config.py` and explain the `history_token_budget` property."

**Turn 1 — model reasons:**
The model sees the request. It needs the content of `config.py`. It emits:

```json
{
  "tool_calls": [
    {
      "name": "read_file",
      "arguments": {"path": "src/saathi/config.py"}
    }
  ]
}
```

**Turn 2 — tool executes, observation appended:**
The `read_file` tool reads the file and returns its content as a string. This is appended as
a `ToolMessage`. The conversation now contains the full file content.

**Turn 3 — model reasons again:**
The model now has the file in context. It reasons: "I can see the `history_token_budget` property.
It returns `int(self.context_window * 0.75)`. No further tool calls are needed." It produces
a final `AIMessage` with an explanation.

The number of tool-call rounds is unbounded — saathi will loop as many times as the model
requests tool calls. In practice, well-designed tasks complete in 1–5 rounds. Pathological cases
(poorly-specified tasks, tools that return errors) can loop longer; the user can interrupt with
Ctrl-C at any time.

### The LangGraph State Graph

LangGraph encodes the ReAct loop as a directed graph with conditional edges.
The graph in `src/saathi/agent/graph.py`:

```python
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)   # the LLM call
builder.add_node("tools", tool_node)   # the tool executor
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
```

`tools_condition` is a LangGraph built-in that inspects the last message in the state:
if it is an `AIMessage` with `tool_calls`, the condition routes to the `"tools"` node;
otherwise it routes to `END`. This single conditional edge implements the entire loop.

The `"tools"` node always routes back to `"agent"` unconditionally. The agent sees the
new `ToolMessage` in its state and either calls more tools or terminates.

```text
START
  │
  ▼
agent ──(has tool_calls)──► tools
  │                            │
  │ (no tool_calls)            │
  ▼                            │
 END                           │
  ▲                            │
  └────────────────────────────┘
         (unconditional)
```

### State Accumulation

LangGraph's `AgentState` accumulates messages across turns. Each node receives the current state,
appends its output messages, and returns the updated state. The checkpointer (SQLite) persists
the full state to disk after each node completes, enabling session resumption.

```python
# src/saathi/agent/state.py (simplified)
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    """Agent state — messages is the accumulated chat history."""
    pass
```

`MessagesState` uses LangGraph's built-in `add_messages` reducer, which appends new messages
to the list rather than replacing it. This is what gives the agent memory within a session.

---

## Structured Outputs

Many agentic tasks require the model to produce output in a predictable structure — a JSON object,
a list of findings, a score. There are two approaches to achieving this: grammar-constrained
decoding and tolerant post-processing. Understanding the difference is critical for building
performant agents.

### JSON Mode and Grammar-Constrained Decoding

Ollama supports a `format="json"` parameter that activates grammar-constrained decoding.
When this parameter is set, llama.cpp uses a context-free grammar to constrain which tokens
are valid at each step of generation. At every position, only tokens that continue a valid
JSON string are in the sampling pool. The result is guaranteed to be syntactically valid JSON.

This sounds ideal. In practice, it is catastrophically slow for structured code review outputs.

### Why Grammar Constraints Are 4–5x Slower

Grammar-constrained decoding imposes a computational overhead at every token generation step.
Instead of sampling from the full vocabulary (e.g., 32,000 tokens for Llama), the runtime must:

1. Evaluate the current partial JSON string against the grammar to determine which terminals
   are valid at this position.
2. Compute the intersection of the grammar's valid tokens with the model's top-k vocabulary.
3. Re-normalise the probability distribution over only the valid tokens.
4. Sample from the constrained distribution.

Steps 1–3 happen at every single token. For a complex grammar like JSON (which requires tracking
brace depth, string escaping, comma placement, and quoted key names), this grammar evaluation
is not trivial. On CPU inference with llama.cpp, the overhead typically reduces throughput by
a factor of 4 to 5.

Empirical measurement from saathi development, running `gemma4:12b` on an NVIDIA RTX 3080
(10 GB VRAM, Q4_K_M weights):

| Mode                      | Tokens/sec | Relative speed |
| --------------------------- | ----------- | ---------------- |
| Free text generation      | ~38 tok/s | 1.0x (baseline)|
| `format="json"` (Ollama)  | ~8 tok/s  | 0.21x          |

A review response that takes 8 seconds in free-text mode takes 40 seconds with JSON mode enabled.
For a multi-reviewer workflow that runs 4 specialist reviewers concurrently, this difference is
the margin between a usable tool and an unusable one.

### Tolerant JSON Parsing: The Saathi Approach

The solution is to let the model generate free text and extract JSON from the output with a
tolerant parser. This is the approach used in `src/saathi/review.py`:

```python
def _extract_json(text: str) -> dict | list | None:
    """Best-effort JSON extraction from an LLM response (tolerates prose/fences)."""
    text = text.strip()
    # Handle markdown code fences (```json ... ```)
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    # Try direct parse first (fast path for clean responses)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, (dict, list)) else None
    except json.JSONDecodeError:
        pass
    # Fallback: scan for the outermost { } or [ ] pair
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, (dict, list)):
                return parsed
    return None
```

This function handles the three most common output patterns from instruction-tuned models:

1. **Clean JSON** — the model outputs exactly the JSON object, nothing else. `json.loads` succeeds
   on the first try.
2. **Fenced JSON** — the model wraps the JSON in a markdown code block (` ```json ... ``` `).
   The fence-stripping pre-process handles this.
3. **Prose + JSON** — the model produces a sentence of explanation followed by the JSON object.
   The brace-scanning fallback finds and extracts the JSON.

Models almost never produce malformed JSON that passes the fence/prose tests but fails `json.loads`.
When they do (typically for very long or nested structures), the system treats the response as
empty (no findings) rather than crashing.

### Pydantic Validation as a Second Gate

Even after tolerant JSON extraction, `review.py` runs each individual finding through a Pydantic
model with lenient validators:

```python
class Finding(BaseModel):
    reviewer: str = ""
    severity: str = "medium"
    confidence: int = 50
    file: str = ""
    line: int | None = None
    issue: str = ""
    suggestion: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> int:
        try:
            n = int(v)
        except (TypeError, ValueError):
            return 50
        return max(0, min(100, n))
```

If the model returns `"confidence": "high"` (a string instead of an integer), the validator
clamps it to 50 rather than raising an exception. If a field is missing, a default is used.
This defensive posture means a partially-formed response still yields usable findings.

---

## The ReAct Pattern in Practice: A Worked Example

The previous sections covered the theory. This section traces a complete multi-step ReAct
execution in saathi to show how tokens, tool calls, and observations interact in reality.

### Task: "Find the retry logic and explain when it triggers"

**System state:**

- `SAATHI_OLLAMA_MODEL=gemma4:12b`
- `SAATHI_TEMPERATURE=0.1`
- `SAATHI_CONTEXT_WINDOW=32768`

**Initial message list** (after system prompt):

```python
messages = [
    SystemMessage(content="You are Saathi..."),  # ~120 tokens
    HumanMessage(content="Find the retry logic and explain when it triggers"),  # ~12 tokens
]
# Estimated total: ~132 tokens / 24576 budget → 0.5% full
```

**Round 1 — Agent node:**
The LLM receives the state. It reasons (internally, not shown) that it needs to search
the codebase for retry-related files. It emits:

```python
AIMessage(
    content="",
    tool_calls=[
        ToolCall(name="search_files", args={"pattern": "retry", "path": "src/saathi"})
    ]
)
```

**Round 1 — Tools node:**
`search_files` executes, returns `["src/saathi/retry.py", "src/saathi/agent/tool_node.py"]`.
A `ToolMessage` is appended. State is now ~160 tokens.

**Round 2 — Agent node:**
Model sees the search results, decides to read `retry.py`:

```python
AIMessage(
    tool_calls=[ToolCall(name="read_file", args={"path": "src/saathi/retry.py"})]
)
```

**Round 2 — Tools node:**
`read_file` returns the content. State grows by the file size (~80 lines × ~10 tokens = ~800 tokens).
Total: ~1000 tokens.

**Round 3 — Agent node:**
The model now has the retry implementation in context. It determines it has enough information
and produces a final answer without tool calls:

```python
AIMessage(
    content="The retry logic in `src/saathi/retry.py` triggers when..."
)
```

`tools_condition` sees no `tool_calls` and routes to `END`. The graph terminates.

Total token consumption for this exchange: ~1100 tokens out of 24,576 available. The remaining
budget is available for the next user turn.

---

## Model Families in 2026

The LLM landscape has evolved rapidly. As of 2026, several model families are well-represented
in Ollama's model library and are suitable for coding assistant tasks.

### Gemma 4 (Google DeepMind)

Gemma 4 is Google's fourth-generation open-weights model family, released in 2025.
It uses a multimodal architecture with alternating local (sliding-window) and global attention
layers, making it more efficient than purely global-attention models at long context lengths.

| Variant       | Parameters | Context window | Quantised VRAM (Q4) | Strength        |
| --------------- | ----------- | ---------------- | ---------------------- | ----------------- |
| gemma4:1b     | 1B        | 32k            | ~1 GB                | Fast edge tasks |
| gemma4:4b     | 4B        | 128k           | ~3 GB                | Balanced        |
| gemma4:12b    | 12B       | 128k           | ~7 GB                | Saathi default  |
| gemma4:27b    | 27B       | 128k           | ~16 GB               | High quality    |

Saathi defaults to `gemma4:12b` for its combination of instruction-following quality,
128k physical context, and compatibility with consumer GPUs (10+ GB VRAM).

### Llama 4 (Meta)

Llama 4, released in early 2025, introduced a Mixture-of-Experts (MoE) architecture to the
Llama family. MoE models activate only a fraction of their parameters for each token,
making them faster per token than dense models of comparable total parameter count.

| Variant           | Active params | Total params | Context window | Notes          |
| ------------------- | -------------- | -------------- | ---------------- | ---------------- |
| llama4-scout:17b  | 17B active   | 109B total   | 10M            | Very long ctx  |
| llama4-maverick   | 17B active   | 400B total   | 1M             | Best quality   |

Llama 4's 10M context window (Scout variant) is remarkable, though useful context in practice
is limited by the lost-in-the-middle problem. For saathi, Llama 4 Scout is a viable alternative
to gemma4:12b on machines with 24+ GB VRAM.

### Mistral (Mistral AI)

Mistral models are known for punching above their parameter count. Mistral Nemo (12B) and
Mistral Small 3.1 (24B) are strong performers on coding benchmarks.

| Variant              | Parameters | Context window | Notes             |
| ---------------------- | ----------- | ---------------- | ------------------- |
| mistral-nemo:12b     | 12B       | 128k           | Multilingual      |
| mistral-small3.1:24b | 24B       | 128k           | Best Mistral coding |

### Phi-4 (Microsoft Research)

Phi-4 models are "small but mighty" — trained on high-quality synthetic data generated from
larger models, they achieve disproportionate quality for their parameter count.

| Variant    | Parameters | Context window | Notes              |
| ------------ | ----------- | ---------------- | -------------------- |
| phi4:14b   | 14B       | 16k            | Strong at math/code |
| phi4-mini  | 3.8B      | 16k            | Laptop-class       |

Phi-4's 16k context window is a limitation for saathi, where tool outputs can quickly
fill the context. Not recommended as saathi's primary model.

### Qwen 3 (Alibaba)

Qwen 3 is Alibaba's third-generation model family. It natively supports 100+ languages and
has strong performance on multilingual coding tasks.

| Variant      | Parameters | Context window | Notes              |
| -------------- | ----------- | ---------------- | -------------------- |
| qwen3:8b     | 8B        | 128k           | Efficient          |
| qwen3:14b    | 14B       | 128k           | Strong coding      |
| qwen3:32b    | 32B       | 128k           | Near-frontier local |

For saathi users working in non-English-speaking environments or on multilingual codebases,
`qwen3:14b` is a strong alternative to `gemma4:12b`.

### Choosing a Model for Saathi

The primary factors in model selection for a coding assistant are:

1. **Instruction following** — does the model reliably use tools, follow format instructions,
   and avoid making up file contents? Gemma 4 and Qwen 3 score well here.
2. **Code quality** — does the model produce syntactically correct, idiomatic code?
   All of the above families are strong.
3. **VRAM headroom** — the chosen `SAATHI_CONTEXT_WINDOW` must fit in available VRAM alongside
   the model weights and KV-cache.
4. **Context window** — for large codebase exploration, a 128k physical context (even if only
   32k is allocated) is preferable.

---

## Local vs Cloud LLMs

Saathi runs entirely locally against an Ollama instance. This is a deliberate architectural
choice, but it is not the only choice. Understanding the tradeoffs helps you decide when to
deviate from the default.

### Privacy

Local inference means your code, prompts, and tool outputs never leave your machine.
For proprietary codebases, regulated industries, or security-sensitive environments, this is
the decisive argument for local deployment. Cloud LLMs require transmitting potentially
sensitive source code to a third-party server.

### Cost

Local inference has zero marginal cost per token. The upfront cost is the GPU hardware.
Cloud inference has no hardware cost but charges per token — typically $0.20–$15.00 per
million tokens depending on the model and provider.

At saathi's usage profile (a few hundred thousand tokens per day for an active developer),
cloud costs for frontier models (GPT-4.1, Claude Sonnet 4) would run $20–$100/month.

### Latency

This is where local inference loses. A well-provisioned local GPU (RTX 3080 or better) generates
30–50 tokens/second with `gemma4:12b` in Q4. Cloud frontier models via API typically deliver
80–150 tokens/second with sub-100ms time-to-first-token when not rate-limited.

For interactive use where the user is reading responses as they stream, 30 tok/s is acceptable.
For automated pipelines that run many reviews or completions in batch, cloud is faster.

### Quality

As of 2026, frontier cloud models (Claude Opus 4, GPT-4.1, Gemini 2.5 Pro) produce meaningfully
better code than any local 12B model. The gap narrows for well-defined coding tasks with good
prompts, but it does not disappear.

For saathi's use case — reading files, making targeted edits, explaining code — the gap is
tolerable. For tasks requiring deep reasoning about complex multi-file interactions, a frontier
cloud model is demonstrably more reliable.

### Reproducibility

Local Ollama inference is highly reproducible. Same model, same temperature, same seed →
same output. Cloud providers version models less transparently and may update model weights
without notice, changing behaviour over time.

### Decision Matrix

| Concern           | Local (Ollama)        | Cloud API              |
| ------------------- | ----------------------- | ------------------------ |
| Data privacy      | Excellent             | Requires trust in provider |
| Marginal cost     | $0/token              | $0.20–$15/M tokens     |
| Hardware cost     | $500–$2000 GPU        | None                   |
| Throughput        | 30–50 tok/s (12B Q4)  | 80–150 tok/s           |
| Output quality    | Good                  | Excellent (frontier)   |
| Offline use       | Yes                   | No                     |
| Reproducibility   | High                  | Medium                 |
| Setup complexity  | Moderate (Ollama)     | Low (API key)          |

> **Note:** Saathi's `ChatOllama` client is OpenAI-compatible, meaning you can point it at
> any OpenAI-compatible endpoint (OpenRouter, LM Studio, vLLM, or the actual OpenAI API) by
> changing `SAATHI_OLLAMA_BASE_URL`. The quality/cost/privacy tradeoff is configurable at
> runtime.

---

## Quantization Deep Dive

Quantization is the most important practical technique for running large models on consumer
hardware. This section gives a more complete treatment than the introductory coverage earlier.

### The GGUF Format

GGUF (GPT-Generated Unified Format) is the file format used by llama.cpp and Ollama to store
quantized models. It is a self-describing binary format that includes:

- The model's hyperparameters (architecture, layer counts, dimensions)
- All tensor data in the specified quantization format
- Tokenizer data (vocabulary, merges, special tokens)
- Metadata (model name, architecture class, training parameters)

A GGUF file is a single file that contains everything needed to run the model. This is why
`ollama pull gemma4:12b` downloads a single multi-gigabyte blob.

### How k-Quants Work

k-quants (introduced in llama.cpp in 2023) quantize weights in blocks of 32 values. For each
block:

1. The 32 floating-point values are scaled so the maximum absolute value maps to the quantization
   range.
2. Each value is rounded to the nearest integer in that range (e.g., 0–15 for 4-bit).
3. The block's scale factor (a 16-bit float) is stored alongside the quantized integers.

The innovation over naive quantization is the per-block scale factor. A single global scale
for the entire weight matrix would force very large absolute values to use a coarse quantization
that destroys precision for small values. Per-block scaling preserves precision where it matters.

### Size Calculation

For a model with `P` billion parameters, the GGUF file size in gigabytes is approximately:

```text
size_GB ≈ P × bits_per_weight / 8
```

Examples for a 12B parameter model:

```text
fp16:   12 × 10⁹ × 16 / 8 = 24 GB
Q8_0:   12 × 10⁹ × 8  / 8 = 12 GB
Q4_K_M: 12 × 10⁹ × 4.5 / 8 ≈ 6.75 GB  (4.5 bits average for K-M variant)
Q2_K:   12 × 10⁹ × 2.5 / 8 ≈ 3.75 GB
```

The VRAM requirement for inference is the model size plus the KV-cache. The KV-cache size:

```text
kv_cache_GB ≈ 2 × num_layers × num_heads × head_dim × context_window × bytes_per_element / 10⁹
```

For `gemma4:12b` with `num_ctx=32768` in fp16 KV-cache:

```text
≈ 2 × 28 × 8 × 256 × 32768 × 2 / 10⁹ ≈ 1.9 GB
```

Total VRAM for saathi's default: ~6.75 + ~1.9 ≈ **8.65 GB**.

> **Note:** Not all GPU memory is equally fast. PCIe bandwidth limits the speed at which weights
> are loaded from VRAM into the shader cores for each forward pass. A GPU with fast HBM memory
> (like H100) runs the same model 3–4x faster than a consumer GPU with GDDR6, even at the
> same VRAM capacity.

### How CPU Inference Works

llama.cpp can run models with layers split across GPU VRAM and CPU RAM. If your GPU has
8 GB VRAM and the model requires 8.65 GB, llama.cpp will place most layers on the GPU
and offload the remainder to CPU RAM. Generation speed degrades proportionally to the
number of layers on CPU, because CPU → GPU data transfer is slow.

Ollama's `--gpu-layers` parameter (or `num_gpu_layers` in the model config) controls
this split. Placing all layers on GPU maximises performance; any CPU offload is a compromise.

---

## Embedding Models

Embedding models convert text into dense numerical vectors (embeddings) that capture semantic
meaning. Two pieces of text that are semantically similar will have embeddings that are close
together in the vector space. This property enables semantic search, document retrieval, and
clustering.

### What Embeddings Are Used For

The primary use case for embeddings in agentic systems is **Retrieval-Augmented Generation (RAG)**:

1. At index time: split your documents into chunks, compute an embedding for each chunk, and store
   them in a vector database.
2. At query time: embed the user's question, find the `k` nearest document chunks by cosine
   similarity, and inject those chunks into the model's context as retrieved context.

RAG is the standard solution when the information a model needs exceeds its context window,
or when you want the model to cite specific source documents.

### Why Saathi Does Not Use Embeddings

Saathi takes a different approach: it reads files directly via tool calls. When the user asks
about a specific module, saathi reads that module. This approach has several advantages:

1. **No index maintenance** — a vector index becomes stale as code changes. Direct file reads
   are always current.
2. **Exact content** — a vector search retrieves approximately relevant chunks; a file read
   retrieves the exact content.
3. **Code navigation** — the `search_files` tool uses string matching and grep, which are more
   reliable for code than embedding similarity (a semantic search for "retry logic" might miss
   a function named `_with_backoff`).

The tradeoff is that saathi consumes more context window per task. For a 500-file codebase,
RAG would be more efficient. For a 50-file codebase (typical saathi target), direct reading
is simpler and more precise.

### Available Embedding Models

If you were to add RAG to saathi, these are the embedding models available in Ollama:

| Model          | Dimensions | Context window | Notes                   |
| ---------------- | ----------- | ---------------- | ------------------------- |
| nomic-embed    | 768       | 8,192          | Good general purpose    |
| mxbai-embed    | 1,024     | 512            | Strong retrieval quality |
| snowflake-arctic-embed | 1,024 | 512       | Strong for code         |
| all-minilm     | 384       | 256            | Very fast, small        |

Embedding models are tiny compared to generative models — `nomic-embed` is around 270 MB.
They run efficiently on CPU and do not require GPU VRAM.

---

## Prompt Engineering

Prompt engineering is the practice of crafting inputs to LLMs to elicit better outputs.
For a coding assistant, it is the primary lever for improving the quality of responses
without changing the model.

### System Prompts

The system prompt establishes the model's persona, constraints, and workflow. Saathi's
base system prompt in `src/saathi/agent/prompts.py` encodes several important practices:

```python
BASE_PROMPT = """\
You are Saathi (साथी) — a coding companion that walks alongside you, not in front.

Your workflow:
1. Think — understand the task and what information you need
2. Use a tool — read a file, run a command, search the codebase
3. Observe — study the tool's output carefully
4. Repeat — keep using tools until you have enough context
5. Answer — give a clear, grounded response

Rules:
- Always read a file before modifying it
- Prefer patch_file over write_file for targeted edits
- Never delete files unless explicitly asked
- Report errors honestly; do not fabricate success
- Cite file paths and line numbers when explaining code
- Prefer small, verifiable steps over large sweeping changes
"""
```

Each rule addresses a failure mode observed in practice:

- "Always read a file before modifying it" — prevents the model from hallucinating file content
  when asked to make a targeted edit.
- "Report errors honestly; do not fabricate success" — prevents the common failure mode of a
  model claiming a command succeeded when the tool returned an error.
- "Prefer small, verifiable steps" — prevents the model from making sweeping changes that
  are hard to review and easy to misapply.

### Mode-Specific Addenda

The system prompt is dynamically extended based on the active mode:

```python
_MODE_ADDENDA: dict[str, str] = {
    "explain": """
MODE: explain
- Read files, never modify them
- Cite exact file path + line number for every claim
- Use plain language; add tables and code blocks where helpful
- When in doubt, say so
""",
    "refactor": """
MODE: refactor
- Use patch_file instead of write_file for targeted changes
- Explain the reason for every modification
- Run tests after changes when a test command is available
- Prefer minimal, focused edits over full rewrites
""",
    "debug": """
MODE: debug
- Reproduce the bug first before attempting a fix
- Read the full stack trace before reading any code
- Apply the smallest possible fix; verify it before reporting done
""",
}
```

These addenda narrow the model's behaviour for the specific task. A model in `explain` mode
that attempts to call `write_file` is violating an explicit constraint — this makes the
violation visible and easy to catch.

### Few-Shot Examples

Few-shot prompting provides examples of the desired input-output pattern directly in the prompt.
For structured outputs, a single well-formed example is often enough to lock the model into
the desired format. In `review.py`, the system prompt for each reviewer includes an example schema:

```python
'Respond with a JSON object of this exact shape:\n'
'{"findings": [{"severity": "high|medium|low", "confidence": 0-100, '
'"file": "path", "line": <number or null>, "issue": "what is wrong", '
'"suggestion": "how to fix"}]}\n'
'If you find nothing, respond with {"findings": []}. Output only JSON.'
```

This inline schema serves as a few-shot example: it shows the model exactly what structure
to produce, including the specific string values for `severity` and the type of `confidence`.

### Chain-of-Thought

Chain-of-thought (CoT) prompting asks the model to reason step by step before giving a final
answer. For coding tasks, this typically manifests as the model first listing what it needs
to do (the "Think" step in saathi's workflow), then acting. The numbered workflow in the base
prompt (`1. Think → 2. Use a tool → 3. Observe → 4. Repeat → 5. Answer`) is a form of
chain-of-thought scaffolding.

### Good vs Bad Prompts

**Bad prompt** (too vague):

```text
Fix the bugs in my code.
```

The model has no context, no file paths, no description of what is broken.
It will either refuse or hallucinate a response.

**Better prompt** (provides context):

```text
There's a bug in src/saathi/compaction.py — when the conversation has exactly
3 user turns, split_for_compaction returns None instead of compacting.
Read the function and fix it.
```

Now the model has a file path, a description of the expected vs actual behaviour,
and a clear scope for the fix.

**Best prompt** (provides reproduction and expected behaviour):

```text
saathi compaction bug: when the conversation has exactly `keep_turns` user turns,
`split_for_compaction` returns None and no compaction happens. Expected behaviour:
compaction should happen when there are MORE than `keep_turns` turns. Read
src/saathi/compaction.py, confirm the off-by-one, and fix it with a minimal change.
```

This prompt specifies the exact module, the nature of the bug, the expected behaviour,
and the constraint (minimal change). It gives the model exactly the information it needs.

### Instruction Following and Model Selection

Not all models follow instructions equally well. Instruction-following quality — the degree
to which a model honours explicit constraints in the system prompt — varies significantly
across model families and sizes. Gemma 4 and Llama 4 are notably strong at instruction
following for their size. Smaller models (under 7B parameters) often fail to maintain
consistent adherence to multi-point instruction sets across a long conversation.

> **Note:** If you observe saathi ignoring the "read before modify" rule or producing output
> that violates the format constraints, the most likely cause is model-level instruction following
> degradation as the context window fills. This is related to the lost-in-the-middle problem
> described in the next section.

---

## The "Lost in the Middle" Problem

The "lost in the middle" problem is one of the most important empirical findings in applied LLM
research. It was documented systematically by Liu et al. (2023) in the paper "Lost in the Middle:
How Language Models Use Long Contexts."

### The Finding

When relevant information is placed at different positions within a long context, LLM performance
on tasks requiring that information varies dramatically by position:

- Information at the **beginning** of the context is recalled with high accuracy.
- Information at the **end** of the context is recalled with high accuracy.
- Information in the **middle** of the context is recalled with significantly lower accuracy,
  with the performance trough deepest for content at the 40–60% position in a long context.

This U-shaped recall curve holds across model sizes and families, though larger models and
models specifically trained for long-context retrieval (like Gemini 1.5 and Llama 4 Scout)
exhibit a shallower trough.

### Why This Happens

The mechanism is not fully understood, but the leading hypothesis relates to the attention
mechanism's effective range during training. Most training sequences are shorter than the
model's maximum context window. When the model encounters a very long sequence, the attention
patterns learned on shorter sequences do not transfer perfectly to extracting information from
the middle of a much longer sequence.

A related factor is the **recency bias** in autoregressive training: the model's objective at
training time is to predict the next token, which requires attending strongly to the most
recent context. Information at the very beginning of a long context is preserved because
it appears in the system prompt and early turns; the model has learned that system prompts
contain important constraints it should maintain throughout.

### Implications for Context Management

For a coding assistant, the practical implications are stark:

1. **Keep the system prompt short and dense** — it occupies the privileged beginning position.
   Saathi's base prompt is deliberately concise.
2. **Place the most relevant tool outputs near the end** — tool results that the model needs
   to directly reference in its answer should arrive as the most recent context.
3. **Do not fill the context window with tangentially relevant content** — a model given a
   full 32k context of loosely related files to answer a specific question will perform worse
   than a model given a focused 4k context with exactly the relevant files.
4. **Compact frequently** — compacting older turns into a summary keeps the actionable content
   near the beginning and end, reducing the amount of "middle" the model must traverse.

### Saathi's Compaction Strategy

Saathi's `src/saathi/compaction.py` implements a two-tier strategy:

**Tier 1 — Budget check:**
At the start of each agent node invocation, the current message list is checked against the
history token budget (75% of `context_window`). If the budget is not exceeded, nothing happens.

**Tier 2 — Summarise-and-compact:**
When the budget is exceeded, the older portion of the conversation (everything before the
last `keep_turns=3` user turns) is summarised by the model into a single `SystemMessage`
and the full transcript of those older turns is replaced by the summary:

```python
async def compact_messages(
    llm: LanguageModelLike,
    messages: list[BaseMessage],
    *,
    keep_turns: int = 3,
) -> list[BaseMessage]:
    split = split_for_compaction(messages, keep_turns)
    if split is None:
        return messages
    older, recent = split

    transcript = "\n".join(
        f"{m.__class__.__name__.replace('Message', '')}: {_text(m)}" for m in older
    )
    response = await llm.ainvoke(
        [
            SystemMessage(content=_SUMMARY_INSTRUCTIONS),
            HumanMessage(content=f"Conversation so far:\n\n{transcript}"),
        ]
    )
    summary_text = _text(response) if isinstance(response, BaseMessage) else str(response)
    summary = SystemMessage(content=f"{_SUMMARY_PREFIX}\n{summary_text}")
    return [summary, *recent]
```

The result is a `[summary, *recent]` list where `summary` is a condensed representation of
everything before the last 3 user turns, and `recent` is the verbatim transcript of the most
recent 3 turns.

### Why This Helps with Lost-in-the-Middle

After compaction, the message list has this structure:

```text
[SystemMessage("You are Saathi..."),     # position: beginning
 SystemMessage("Summary of earlier..."), # position: just after beginning
 HumanMessage(...),                      # recent turn -2
 AIMessage(...),                         # recent turn -2 response
 ToolMessage(...),                       # recent turn -2 tool result
 HumanMessage(...),                      # recent turn -1
 AIMessage(...),                         # recent turn -1 response
 HumanMessage(...),                      # current turn (end position)
]
```

The summary occupies the beginning (high recall) position. The current and immediately
preceding turns occupy the end (high recall) position. The middle of the context — where
recall is weakest — is now occupied only by a compact summary rather than raw verbose
transcripts that would demand verbatim recall.

This is not a perfect solution. The summary itself is a lossy compression — details not
captured in the summary are permanently lost. The `_SUMMARY_INSTRUCTIONS` prompt is designed
to preserve the most actionable details:

```python
_SUMMARY_INSTRUCTIONS = (
    "You are compacting a coding-assistant conversation to save context window. "
    "Write a concise summary capturing: the user's goals, key decisions, files "
    "read or modified, important findings, and any unresolved threads. Preserve "
    "concrete details a developer would need to continue. Output only the summary."
)
```

"Files read or modified" and "concrete details a developer would need to continue" are the
key phrases. A summary that says "the agent read some files" is useless; one that says
"the agent read `src/saathi/config.py` and found that `context_window` defaults to 32768"
is actionable.

### Practical Recommendations

Given the lost-in-the-middle constraint, here are practical guidelines for getting the best
from saathi on large tasks:

**Break large tasks into focused sub-tasks.** Instead of asking "refactor the entire agent
module," ask "read `src/saathi/agent/nodes.py` and suggest improvements to the error handling."
Focused tasks produce focused contexts.

**Use mode selection.** Starting a session with `saathi --mode explain` restricts the agent
to read-only operations, which keeps context consumption lower for exploratory sessions.

**Reset sessions for new topics.** When you switch from one task to a completely different
task in the same saathi session, consider restarting the session rather than carrying
forward an unrelated conversation history. Old, summarised context from an unrelated task
consumes budget and can confuse the model.

**Monitor context usage.** Saathi logs compaction events. If you see frequent compaction
(multiple times within a single task), it indicates that tool outputs are large and you
may benefit from a larger `SAATHI_CONTEXT_WINDOW` or a more powerful GPU.

---

## Summary

This chapter has built a complete mental model of the LLM ecosystem as it pertains to
saathi and similar coding-assistant tools.

**Large language models** are transformers that predict the next token in a sequence. Their
behaviour is encoded in billions of floating-point weights. Quantization (Q4_K_M being the
practical sweet spot) makes it feasible to run 12B parameter models on consumer GPUs.

**Tokenization** converts text to integer sequences. The 4 chars/token heuristic is a useful
approximation. Token count drives both cloud API cost and context window consumption.

**Context windows** define the model's working memory. Saathi defaults to 32k tokens,
reserving 75% for conversation history and triggering compaction when that budget is exceeded.

**Inference parameters** — temperature, top-p, num_predict, num_ctx — control the sampling
process. Saathi uses `temperature=0.1` for deterministic, consistent coding outputs.

**Chat message formats** follow the system/user/assistant/tool role convention. Message
ordering is not arbitrary; tool messages must immediately follow the tool call they answer.

**Tool calling** lets models emit structured requests for external function execution.
Parallel tool execution (up to `max_parallel_tools=8`) allows concurrent file reads and
searches.

**The ReAct loop** (Reason → Act → Observe → repeat) is the fundamental agent pattern.
LangGraph encodes it as a two-node graph with a conditional edge that loops back to the
agent as long as tool calls are present.

**Structured outputs** should use tolerant JSON parsing rather than grammar-constrained
decoding. Grammar constraints (`format="json"`) reduce generation throughput by 4–5x
on local models.

**Model families** — Gemma 4, Llama 4, Mistral, Phi-4, Qwen 3 — each have different
parameter counts, context windows, and strengths. Saathi defaults to `gemma4:12b`.

**Local vs cloud LLMs** trade privacy and zero marginal cost against quality and throughput.
Saathi is architecturally compatible with any OpenAI-compatible endpoint.

**Quantization** enables running large models on small GPUs. The GGUF format is the standard
for llama.cpp and Ollama.

**Embedding models** are used for RAG. Saathi avoids RAG in favour of direct file reading,
which is more precise for code navigation.

**Prompt engineering** — system prompts, mode addenda, few-shot examples, chain-of-thought
scaffolding — is the primary lever for improving agent quality without changing the model.

**Lost in the middle** is the empirical finding that LLMs recall information at the beginning
and end of long contexts far better than information in the middle. Saathi's compaction
strategy is designed to keep actionable content in the high-recall positions.

---

## Further Reading

The following resources provide deeper treatment of the topics covered in this chapter.

**Transformer architecture:**

- Vaswani et al. (2017). "Attention Is All You Need." *NeurIPS 2017.*
  https://arxiv.org/abs/1706.03762
- Illustrated Transformer, Jay Alammar.
  https://jalammar.github.io/illustrated-transformer/

**Tokenization:**

- OpenAI tiktoken library. https://github.com/openai/tiktoken
- SentencePiece tokenizer. https://github.com/google/sentencepiece
- Karpathy's MinBPE (educational BPE implementation).
  https://github.com/karpathy/minbpe

**Quantization and GGUF:**

- llama.cpp project. https://github.com/ggerganov/llama.cpp
- GGUF format specification.
  https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- TheBloke's quantization explanations.
  https://huggingface.co/TheBloke

**ReAct and agents:**

- Yao et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models."
  https://arxiv.org/abs/2210.03629
- LangGraph documentation. https://langchain-ai.github.io/langgraph/

**Lost in the middle:**

- Liu et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts."
  https://arxiv.org/abs/2307.03172

**Structured outputs and grammar constraints:**

- llama.cpp grammar sampling documentation.
  https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
- Ollama structured outputs guide.
  https://ollama.com/blog/structured-outputs

**Prompt engineering:**

- Anthropic's prompt engineering guide.
  https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
- OpenAI prompt engineering guide.
  https://platform.openai.com/docs/guides/prompt-engineering

**Model families (2025–2026):**

- Gemma 4 technical report. https://ai.google.dev/gemma
- Llama 4 release blog. https://ai.meta.com/blog/llama-4/
- Mistral model documentation. https://docs.mistral.ai/
- Qwen 3 technical report. https://qwenlm.github.io/blog/qwen3/
