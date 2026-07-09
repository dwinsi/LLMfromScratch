# Chapter 18 — LangSmith: Observability, Tracing, and Debugging

> "If you can't measure it, you can't improve it."
>
> — Peter Drucker

---

## 18.1 The Observability Problem for LLM Applications

Traditional application performance monitoring (APM) tools—Datadog, New Relic, Dynatrace, AppDynamics—were designed for a world of HTTP requests and database queries. They understand latency, throughput, error rates, and resource utilisation. They are excellent at answering questions like: "Which API endpoint is slow?" and "How many database queries does the checkout page make?"

They are not equipped to answer the questions that matter for AI applications:

- "What prompt did the model actually receive for this request?"
- "Why did the agent call `read_file` three times in a row?"
- "How many tokens did the system prompt consume?"
- "Which tool call accounted for 80% of the latency on this turn?"
- "The agent's answer changed between deploys—what changed in the prompt?"
- "How does the model's output quality vary with temperature?"
- "What is the 95th-percentile response latency broken down by tool call count?"

These questions require understanding the internal structure of an LLM call: the prompt, the completion, the tools, the chain of reasoning. A distributed trace that records "HTTP POST /chat → 3.2s" is not useful. A trace that records "agent node: 1.1s (512 input tokens, 98 output tokens) → tool node: read_file (0.2s) → agent node: 0.9s (680 input tokens, 210 output tokens)" is what you need.

### The Unique Challenges of LLM Observability

**Non-determinism.** Given the same input, an LLM may produce different outputs. This makes debugging harder: you cannot simply replay a request and compare the output. You need to capture the exact prompt, parameters (temperature, top-p), and model version.

**Multi-step reasoning.** A single user request might trigger 5 LLM calls and 12 tool calls, connected in a complex graph. Understanding which step produced a bad output requires tracing the entire chain.

**Token economics.** LLM costs are proportional to token usage. Without instrumentation, you have no way to know whether your costs are dominated by long system prompts, verbose tool outputs, or repetitive context.

**Prompt fragility.** A small change to a system prompt can dramatically affect model behaviour. Without version-controlled, traced prompts, debugging regressions is guesswork.

**Tool call attribution.** When the agent behaves unexpectedly, was it the LLM that made the wrong decision, or did a tool return incorrect data? Without per-step tracing, you cannot tell.

**Evaluation regression.** As you improve your agent, you need to ensure that changes do not break previously working scenarios. This requires a test suite, a dataset, and a way to run automated evaluations.

LangSmith is purpose-built to address all of these challenges.

---

## 18.2 What Is LangSmith?

LangSmith is a platform for tracing, debugging, and evaluating LLM applications built with LangChain and LangGraph. It was developed by LangChain, Inc., the company behind the LangChain and LangGraph open-source libraries.

Despite the name, LangSmith is not an Anthropic product. It is a commercial product built by LangChain, Inc. (The confusion is understandable given the naming overlap.) LangSmith integrates so tightly with LangChain/LangGraph that it feels like a first-party feature—it requires minimal setup and provides deep insight into every aspect of your LangGraph runs.

### What LangSmith Provides

1. **Automatic tracing.** Set two environment variables and every LangChain/LangGraph run is traced. No code changes required.
2. **Run explorer.** A web UI where you can browse all runs, filter by project, date, or metadata, and inspect individual traces.
3. **Trace detail view.** For each run, see the full tree of nodes/tools with latency, token counts, and input/output at each step.
4. **Prompt viewer.** See the exact prompt sent to the model—system message, conversation history, tool definitions—rendered in a human-readable format.
5. **Output viewer.** See the model's exact completion, including any tool call JSON.
6. **Datasets.** Collections of (input, expected output) examples for evaluation.
7. **Evaluations.** Run automated evaluations against datasets to measure model quality.
8. **Comparison view.** Compare two runs side by side—useful when A/B testing prompts or models.
9. **Feedback.** Annotate runs with thumbs-up/thumbs-down or custom scores for supervised fine-tuning signals.
10. **Dashboards.** Aggregate metrics: average latency, total tokens, error rate, score distributions.

### Pricing

LangSmith offers:

- **Developer (free tier)**: 5,000 traces/month, 14-day data retention. Sufficient for development and experimentation.
- **Plus**: Higher limits, longer retention.
- **Enterprise**: Unlimited, self-hosted option available, SLA guarantees.

For saathi development work, the free tier is adequate.

---

## 18.3 Setting Up LangSmith

### Step 1: Create an Account

Go to [smith.langchain.com](https://smith.langchain.com) and sign up. You will need to create an account and a project.

### Step 2: Generate an API Key

In the LangSmith UI:

1. Click your username → **Settings**
2. Navigate to **API Keys**
3. Click **Create API Key**
4. Copy the key (it starts with `ls__`)

### Step 3: Configure Environment Variables

LangSmith uses a specific set of environment variables (not the `SAATHI_` prefix—these are LangChain conventions):

```bash
# Enable tracing.
LANGCHAIN_TRACING_V2=true

# Your LangSmith API key.
LANGCHAIN_API_KEY=ls__your_key_here

# The project name in LangSmith (creates if it doesn't exist).
LANGCHAIN_PROJECT=saathi-langgraph

# Optional: custom LangSmith endpoint (for self-hosted).
# LANGCHAIN_ENDPOINT=https://your-langsmith-instance.com
```

Add these to your `.env` file:

```bash
# .env (excerpt)

# ─────────────────────────────────────────────────────────────────────────────
# LangSmith tracing (optional)
# ─────────────────────────────────────────────────────────────────────────────

# Set to true to enable LangSmith tracing.
# Get an API key at: https://smith.langchain.com
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_PROJECT=saathi-langgraph
```

That is all. No code changes. No library to import. No callback to register. The LangChain SDK reads these environment variables automatically and sends traces to LangSmith in a background thread.

### Step 4: Verify

Run saathi and perform a task:

```bash
saathi --print "List all Python files"
```

Then open [smith.langchain.com](https://smith.langchain.com), navigate to your project (`saathi-langgraph`), and you should see a trace appear within a few seconds.

### Installing the LangSmith SDK (Optional)

The core tracing works without installing `langsmith` explicitly—it is bundled with `langchain-core`. However, for evaluation and dataset features, install it directly:

```bash
pip install langsmith
# or add to pyproject.toml:
# langsmith = "^0.1"
```

---

## 18.4 What Gets Traced Automatically

When `LANGCHAIN_TRACING_V2=true` is set, LangChain automatically traces:

### LangGraph Graph Runs

Every call to `graph.ainvoke()` or `graph.astream()` creates a top-level trace (called a "run" in LangSmith). The trace includes:

- The full input state
- The full output state
- The total elapsed time
- The total token count across all nodes
- A status (success / error)

### Node Executions

Each node in the LangGraph graph creates a child span within the run:

- **`agent` node**: Records the LLM call. Input: the full message list (system prompt + conversation history + tool definitions). Output: the AIMessage (text + tool call decisions). Token counts: input tokens, output tokens. Latency.
- **`tools` node**: Records each tool execution. Input: tool name + arguments. Output: tool result. Latency.

### LLM Calls

Within the agent node, the individual LLM call is traced:

- The exact messages array sent to the model
- The temperature, max_tokens, and other parameters
- The model's response
- Token counts (prompt tokens, completion tokens)
- Latency to first token (for streaming) and total latency

### Tool Calls

When the agent decides to call a tool:

- Tool name
- Tool arguments (the JSON the model produced)
- Tool result (the string returned by the tool function)
- Latency

### A Complete Trace Structure

For a saathi turn where the agent reads a file and summarises it, the trace looks like:

```flow
[Run] saathi session turn                     3.2s  |  820 in  |  198 out
  └─ [Graph] StateGraph.ainvoke               3.2s
       ├─ [Node] agent                        1.1s  |  512 in  |   32 out
       │    └─ [LLM] ChatOllama               1.1s
       │         ├─ Input:  [SystemMessage, HumanMessage × 4]
       │         └─ Output: AIMessage (tool_call: read_file)
       ├─ [Node] tools                        0.2s
       │    └─ [Tool] read_file               0.2s
       │         ├─ Args:   {"path": "src/saathi/graph.py"}
       │         └─ Result: "# src/saathi/graph.py\n..."
       └─ [Node] agent                        1.9s  |  308 in  |  166 out
            └─ [LLM] ChatOllama               1.9s
                 ├─ Input:  [SystemMessage, HumanMessage × 4, AIMessage, ToolMessage]
                 └─ Output: AIMessage (content: "The graph.py file defines...")
```

This tree structure—with latencies and token counts at each node—is exactly what LangSmith renders in its trace detail view.

---

## 18.5 The LangSmith UI

### Traces List

The main view shows all runs for your project in reverse chronological order. Each row shows:

- Run name (usually the first few words of the input)
- Status (✓ success, ✗ error)
- Start time
- Latency
- Token count (total input + output)
- Feedback score (if annotated)

You can filter by:

- Date range
- Status (success only, error only)
- Tags
- Feedback scores
- Latency range
- Token count range

### Trace Detail View

Click a run to open the trace detail. The left panel shows the tree of nodes/spans. The right panel shows the selected node's details.

**Tree navigation**: Click any node to see its details. Nodes are colour-coded by type (LLM calls in blue, tool calls in purple, custom spans in green).

**Timeline view**: A Gantt-chart-style timeline shows which operations ran in parallel and which were sequential. For saathi (single-threaded turns), the timeline is a straight line. For a parallel tool execution scenario, you would see parallel bars.

### Prompt Viewer

Click an LLM node and select the "Prompt" tab. LangSmith renders the full messages array in a human-readable format:

```text
─ System ──────────────────────────────────────────────────────────────────────
You are saathi, an agentic coding assistant. You have access to tools for
reading files, running shell commands, applying patches, and browsing the web.

Always use tools to understand the codebase before answering.
...

─ Human ───────────────────────────────────────────────────────────────────────
Explain the build_graph function in graph.py.

─ Assistant ───────────────────────────────────────────────────────────────────
[Tool call: read_file(path="src/saathi/graph.py")]

─ Tool ────────────────────────────────────────────────────────────────────────
[Result of read_file]
# src/saathi/graph.py
from langgraph.graph import StateGraph, END
...
```

This view is invaluable for debugging. When the model produces an unexpected response, opening the prompt viewer often immediately reveals why: a misformatted tool result, a truncated history, an instruction in the system prompt that conflicts with the user's request.

### Output Viewer

The "Output" tab shows the model's raw completion—including tool call JSON:

```json
{
  "content": "",
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": "{\"path\": \"src/saathi/graph.py\"}"
      }
    }
  ]
}
```

Seeing the raw tool call JSON is useful when debugging tool argument parsing issues (e.g., why did the model pass `path: "graph.py"` instead of `path: "src/saathi/graph.py"`?).

### Latency Breakdown

The trace detail shows latency at each level. For a 3.2-second turn, you might see:

- Agent node 1: 1.1s (LLM decision)
- Tools node: 0.2s (file read)
- Agent node 2: 1.9s (LLM synthesis)

If the LLM calls dominate (as they usually do), optimisation should focus on reducing token counts (shorter prompts, more aggressive compaction). If the tools node dominates, the tools are slow (network calls, heavy computation).

---

## 18.6 Debugging with LangSmith

LangSmith's most immediate practical value is debugging unexpected agent behaviour. Here is the workflow.

### Scenario: The Agent Calls the Wrong Tool

You ask: "Add a docstring to the `build_graph` function."

Expected behaviour: The agent reads `graph.py`, generates a docstring, and applies it with `apply_patch` or `write_file`.

Actual behaviour: The agent calls `run_shell("cat graph.py")` instead of `read_file`, then writes a docstring that does not match the actual function signature.

**Without LangSmith**: You guess. Maybe the system prompt is ambiguous? Maybe `read_file` is not in the tool list? Maybe the model's temperature is too high?

**With LangSmith**: Open the trace. Click the first agent node. Open the Prompt tab. Scroll to the tool definitions section. You immediately see: `read_file` is listed, but its description says "Read the contents of a file at a given path" while `run_shell` says "Run a shell command and return its output". The model preferred `run_shell` because your system prompt says "prefer shell tools when possible"—a leftover from an earlier version.

One look at the prompt, problem found. Without LangSmith, this could have taken an hour of print-debugging.

### Scenario: Unexpectedly Long Response Time

A turn that normally takes 2 seconds suddenly takes 12.

**Without LangSmith**: Add timing logs, redeploy, reproduce.

**With LangSmith**: Open the trace. The timeline shows the tools node took 10 seconds. Click it. The `run_shell` tool call shows: `run_shell("pytest tests/ -v")`. The test suite ran and took 10 seconds. The agent decided to run tests as part of its task—correct behaviour, but unexpected. The fix: add a note to the system prompt that running tests is optional unless explicitly requested.

### Scenario: Context Window Overflow

After a long session, the agent's responses become nonsensical or it starts ignoring recent instructions.

**With LangSmith**: Look at the token counts in the trace. The input token count for recent turns is near or above `context_window`. LangSmith will show you exactly which messages are filling the context: a verbose tool result (e.g., a 3000-token file listing), repeated system prompt content, or accumulated conversation history.

The fix: tighten tool outputs (truncate file listings), trigger `/compact` earlier, or increase `SAATHI_CONTEXT_WINDOW`.

### Scenario: Regression After Prompt Change

You modified the system prompt to improve code review quality. Now the agent's `/commit` command produces bad commit messages.

**With LangSmith**:

1. Filter traces to before and after the deploy.
2. Find a `/commit` trace from before and one from after.
3. Use LangSmith's comparison view to see the prompts side by side.
4. The diff shows the new system prompt section on code review style inadvertently overrode the commit message format instructions.

---

## 18.7 Token Cost Tracking

LangSmith aggregates token usage across all runs in your project. The dashboard shows:

- **Total tokens** by day/week/month
- **Breakdown** by model (if you use multiple models)
- **Average tokens per run** (input vs. output)
- **Token count distribution** (histogram)

### Why This Matters Even with Local Ollama

With Ollama, there is no per-token cost. But the token usage data is still valuable:

**Capacity planning.** If you deploy saathi on a shared server, token throughput determines how many concurrent users you can support before the GPU becomes saturated. LangSmith's data lets you calculate: "At current usage, I need X GPU-hours per day."

**Context window headroom.** Tracking average input token counts tells you how close you are to the context window limit. If average input tokens are 6500 and your context window is 8096, you have 1596 tokens of headroom before compaction kicks in. LangSmith makes this visible without instrumentation code.

**Model selection.** When comparing a larger model (slower, more capable) with a smaller model (faster, less capable), token counts alone do not capture the tradeoff. LangSmith lets you compare total tokens AND response quality (via feedback scores) in the same view.

### Cloud LLM Cost Projections

If you add support for cloud LLMs (OpenAI, Anthropic, etc.) in the future, LangSmith's token data becomes directly financial. At OpenAI pricing (approximate, as of 2026):

- GPT-4o: $5/1M input tokens, $15/1M output tokens
- GPT-4o-mini: $0.15/1M input tokens, $0.60/1M output tokens

If LangSmith shows you are averaging 2000 input tokens and 300 output tokens per turn, at 100 turns/day:

```text
Daily cost (GPT-4o):     (2000 × 100 × $5/1M) + (300 × 100 × $15/1M)
                       = $1.00 + $0.45 = $1.45/day = ~$44/month
Daily cost (GPT-4o-mini): (2000 × 100 × $0.15/1M) + (300 × 100 × $0.60/1M)
                       = $0.03 + $0.018 = $0.048/day = ~$1.44/month
```

This kind of calculation is only possible if you have token usage data. LangSmith provides it.

---

## 18.8 Run Metadata — Enriching Traces

By default, LangSmith captures what you said and what the model said. You can enrich traces with metadata to make filtering and analysis more powerful.

### Adding Metadata to a Run

Pass a `config` dict with a `"metadata"` key when calling `ainvoke` or `astream`:

```python
config = {
    "configurable": {
        "thread_id": session_id,
    },
    "metadata": {
        "user_id": "ashwin.kumar",
        "session_id": session_id,
        "project": "saathi-langgraph",
        "saathi_version": "0.1.0",
        "command": "/code-review",
        "model": cfg.model,
        "context_window": cfg.context_window,
    },
    "tags": ["interactive-session", cfg.model],
}

result = await graph.ainvoke(state, config=config)
```

With this metadata, you can filter LangSmith to see:

- All runs from a specific user
- All runs for a specific session
- All `/code-review` runs
- All runs with `qwen2.5:14b`

### Tags

Tags are simple string labels. They are visible in the traces list and filterable. Useful tag strategies:

```python
tags = [
    cfg.model,                    # "qwen2.5:14b"
    "interactive" if interactive else "print-mode",
    "debug" if cfg.debug else "production",
]
```

### Run Names

By default, LangSmith uses the graph name as the run name. You can override it:

```python
config = {
    "run_name": f"saathi: {task[:60]}",
    "metadata": {...},
}
```

This makes the traces list much more readable: instead of "StateGraph.ainvoke" for every run, you see "saathi: Explain the build_graph function" and "saathi: Add docstring to commit()".

---

## 18.9 LangSmith Datasets and Evaluation

Once you have traces, you can promote individual examples into a **dataset** for systematic evaluation.

### Creating a Dataset

A dataset is a named collection of (input, expected output) examples. You create one in the LangSmith UI:

1. Find a trace where the agent gave a great answer.
2. Click **Add to Dataset**.
3. Create a new dataset (e.g., "saathi-qa") or add to an existing one.
4. LangSmith captures the input state and the output state as an example.

Alternatively, create a dataset programmatically:

```python
from langsmith import Client

client = Client()

dataset = client.create_dataset(
    dataset_name="saathi-qa",
    description="Question-answer pairs for saathi regression testing",
)

# Add examples.
client.create_examples(
    inputs=[
        {"messages": [{"type": "human", "content": "What does build_graph() do?"}]},
        {"messages": [{"type": "human", "content": "List all Python files"}]},
    ],
    outputs=[
        {"answer": "build_graph() constructs the LangGraph StateGraph..."},
        {"answer": "The Python files in this project are: ..."},
    ],
    dataset_id=dataset.id,
)
```

### Running an Evaluation

An evaluation runs your agent against all examples in a dataset and scores the results:

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

def run_saathi(inputs: dict) -> dict:
    """Run saathi on the input and return the output."""
    import asyncio
    from saathi.graph import build_graph
    from saathi.config import settings
    from langchain_core.messages import HumanMessage

    messages = inputs["messages"]
    lc_messages = [HumanMessage(content=m["content"]) for m in messages]
    state = {"messages": lc_messages, "model": settings.model}

    graph = build_graph(settings)
    result = asyncio.run(graph.ainvoke(state))

    # Extract the final response.
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.content:
            return {"response": msg.content}

    return {"response": ""}


results = evaluate(
    run_saathi,
    data="saathi-qa",
    evaluators=["correctness"],  # built-in evaluator
    experiment_prefix="saathi-baseline",
    metadata={"model": "qwen2.5:14b", "version": "0.1.0"},
)

print(results.to_pandas())
```

The evaluation runs each example through `run_saathi`, collects the output, and passes input/output pairs to the evaluators. Results are stored in LangSmith and visible in the **Experiments** tab.

---

## 18.10 Custom Evaluators

LangSmith's built-in evaluators (correctness, relevance, conciseness) use an LLM judge to score outputs. For domain-specific evaluation, write your own.

### The Evaluator Interface

A custom evaluator is a Python function that accepts an `EvaluationResult` (containing the input, actual output, and expected output) and returns a score:

```python
from langsmith.schemas import Run, Example
from langsmith.evaluation import EvaluationResult


def contains_file_path_evaluator(run: Run, example: Example) -> EvaluationResult:
    """Evaluate whether the response mentions a file path.

    For tasks that require file inspection, the response should reference
    at least one file path. This is a simple heuristic evaluator.

    Args:
        run: The LangSmith run (contains the actual output).
        example: The dataset example (contains the expected output).

    Returns:
        An EvaluationResult with a score of 1.0 (pass) or 0.0 (fail).
    """
    import re

    response = run.outputs.get("response", "")

    # Look for file path patterns.
    file_path_pattern = re.compile(
        r'(?:src/|tests/|\.saathi/)\S+\.py|`[^`]+\.py`'
    )

    score = 1.0 if file_path_pattern.search(response) else 0.0
    comment = (
        "Response mentions a file path." if score == 1.0
        else "Response does not mention any file path."
    )

    return EvaluationResult(
        key="mentions_file_path",
        score=score,
        comment=comment,
    )


def response_length_evaluator(run: Run, example: Example) -> EvaluationResult:
    """Evaluate that the response is neither too short nor too long.

    A response under 50 characters is probably an error.
    A response over 2000 characters is probably verbose.

    Args:
        run: The LangSmith run.
        example: The dataset example.

    Returns:
        An EvaluationResult with a score between 0.0 and 1.0.
    """
    response = run.outputs.get("response", "")
    length = len(response)

    if length < 50:
        score = 0.0
        comment = f"Response too short ({length} chars)"
    elif length > 2000:
        score = 0.5
        comment = f"Response may be verbose ({length} chars)"
    else:
        score = 1.0
        comment = f"Response length OK ({length} chars)"

    return EvaluationResult(
        key="response_length",
        score=score,
        comment=comment,
    )


def llm_correctness_evaluator(run: Run, example: Example) -> EvaluationResult:
    """Use an LLM to judge whether the response correctly answers the question.

    This is an "LLM-as-judge" evaluator. It uses a fast model (llama3.2:3b)
    to evaluate the response quality, avoiding the cost of a more expensive
    cloud model for bulk evaluation.

    Args:
        run: The LangSmith run.
        example: The dataset example.

    Returns:
        An EvaluationResult with a score between 0.0 and 1.0.
    """
    import json
    import subprocess

    question = example.inputs.get("messages", [{}])[-1].get("content", "")
    expected = example.outputs.get("answer", "")
    actual = run.outputs.get("response", "")

    prompt = f"""You are evaluating an AI assistant's answer to a question.

Question: {question}
Expected answer (reference): {expected}
Actual answer: {actual}

Rate the actual answer's correctness compared to the expected answer.
Respond with a JSON object: {{"score": <0.0 to 1.0>, "reason": "<brief explanation>"}}
Score 1.0 = correct and complete, 0.5 = partially correct, 0.0 = wrong or missing.
"""

    # Use local Ollama for evaluation.
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.2:3b", prompt],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip()
        # Extract JSON from output.
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            score = float(data.get("score", 0.5))
            reason = data.get("reason", "")
            return EvaluationResult(
                key="llm_correctness",
                score=score,
                comment=reason,
            )
    except Exception as exc:
        pass

    return EvaluationResult(
        key="llm_correctness",
        score=0.5,
        comment="Could not evaluate (LLM judge unavailable)",
    )
```

### Running Evaluations with Custom Evaluators

```python
results = evaluate(
    run_saathi,
    data="saathi-qa",
    evaluators=[
        contains_file_path_evaluator,
        response_length_evaluator,
        llm_correctness_evaluator,
    ],
    experiment_prefix="saathi-custom-eval",
    max_concurrency=2,  # run 2 examples in parallel
)
```

Custom evaluators give you domain-specific quality metrics that generic built-in evaluators cannot provide.

---

## 18.11 The `@traceable` Decorator — Tracing Arbitrary Functions

LangSmith's automatic tracing captures LangChain/LangGraph operations. For custom Python functions that are not LangChain components—but are important steps in your pipeline—the `@traceable` decorator adds them to the trace tree.

### Basic Usage

```python
from langsmith import traceable


@traceable(name="compact_messages", run_type="chain")
async def compact_messages(messages: list, cfg) -> list:
    """Summarise and compress the message history.

    This is traced as a custom span in LangSmith, so you can see
    when compaction happens and how many tokens were saved.
    """
    # ... compaction logic ...
    return compressed_messages
```

When `compact_messages` is called during a traced run, it appears as a child span:

```flow
[Run] saathi session turn
  └─ [Graph] StateGraph.ainvoke
       ├─ [Node] agent                (LLM call)
       ├─ [Chain] compact_messages    ← custom span
       │    ├─ Input:  {"messages": [...42 messages...]}
       │    └─ Output: [...8 messages...]
       └─ [Node] agent                (LLM call with compacted history)
```

You can see that compaction reduced 42 messages to 8, and see the latency of the compaction operation itself.

### Tracing a Code Review Function

```python
@traceable(name="run_code_review", run_type="chain")
async def run_code_review(diff: str, cfg) -> str:
    """Run a code review on the provided diff.

    Traced so we can see:
    - How long the code review takes end-to-end
    - How large the diff input is (tokens)
    - The quality of the review output
    """
    # ... review logic ...
    return review_text
```

### `run_type` Options

The `run_type` parameter controls how LangSmith categorises and displays the span:

| `run_type` | When to use |
| ------------ | ------------- |
| `"llm"` | Direct LLM calls (usually set automatically by LangChain) |
| `"chain"` | Multi-step logic, custom pipelines |
| `"tool"` | Tool calls, external API calls |
| `"retriever"` | RAG document retrieval operations |
| `"embedding"` | Embedding generation |
| `"parser"` | Output parsing, structured extraction |

For most custom saathi functions, `"chain"` is appropriate.

### Tracing with Metadata

```python
@traceable(
    name="memory_load",
    run_type="retriever",
    metadata={"source": "filesystem"},
)
async def load_memory(memory_dir: Path) -> str:
    """Load persistent memory from disk."""
    # ...
```

---

## 18.12 LangSmith for saathi — What You Would See

If you enable LangSmith tracing on saathi today, here is what appears in the UI for a typical interactive session:

### Turn 1: "Explain the build_graph function"

```flow
[Run] saathi: Explain the build_graph function     2.1s  |  634 in  |  187 out

  [Node] agent                                     1.8s  |  220 in  |   34 out
    [LLM] ChatOllama (qwen2.5:14b)                 1.8s
      Input:
        [system]: You are saathi, an agentic coding assistant...
        [human]:  Explain the build_graph function
      Output:
        [ai]: (tool call: read_file(path="src/saathi/graph.py"))

  [Node] tools                                     0.1s
    [Tool] read_file                               0.1s
      Args:   {"path": "src/saathi/graph.py"}
      Result: "# src/saathi/graph.py\n..."

  [Node] agent                                     0.2s  |  414 in  |  153 out
    [LLM] ChatOllama (qwen2.5:14b)                 0.2s
      Input:
        [system]: You are saathi...
        [human]:  Explain the build_graph function
        [ai]:     (tool call: read_file)
        [tool]:   "# src/saathi/graph.py\n..."
      Output:
        [ai]: "The `build_graph` function in `graph.py` constructs..."
```

### Turn 2: "/compact triggered automatically"

```flow
[Run] saathi: (compact)                            0.8s  |  1,240 in  |  180 out

  [Chain] compact_messages                         0.1s
    Input:  {"messages": [...24 messages..., 6,100 tokens estimated]}
    Output: {"messages": [...4 messages..., 720 tokens estimated]}

  [Node] agent                                     0.7s  |  1,240 in  |  180 out
    [LLM] ChatOllama (qwen2.5:14b)                 0.7s
      (summarisation prompt)
```

### Turn 3: "/code-review"

```flow
[Run] saathi: /code-review                         4.5s  |  2,100 in  |  420 out

  [Node] agent                                     1.2s
    [LLM]: "I'll review the current diff."
    (tool calls: run_shell(git diff), read_file × 3)

  [Node] tools                                     0.4s
    [Tool] run_shell        0.1s   Args: {"command": "git diff HEAD"}
    [Tool] read_file        0.1s   Args: {"path": "src/saathi/cli.py"}
    [Tool] read_file        0.1s   Args: {"path": "src/saathi/display.py"}
    [Tool] read_file        0.1s   Args: {"path": "tests/test_display.py"}

  [Node] agent                                     2.9s
    [LLM]: "Here is my code review of the current diff:..."
```

This level of visibility—which tools were called, what their inputs and outputs were, how long each step took, how many tokens each LLM call consumed—is what transforms debugging from guesswork to observation.

---

## 18.13 Privacy Considerations

### What Data Does LangSmith Receive?

When `LANGCHAIN_TRACING_V2=true`, every LLM call sends to LangSmith:

- The full message array, including the system prompt, conversation history, and tool definitions.
- The model's full response.
- Token counts and latency.
- Any metadata you add.

**If your conversation includes sensitive information** (e.g., you paste in a secrets file by accident, or discuss proprietary code), that content is transmitted to LangSmith's servers.

### saathi and Local Ollama

With Ollama, your prompts and responses never leave your machine in transit to the LLM. Enabling LangSmith tracing does send them to LangSmith's cloud. This is opt-in (you must set `LANGCHAIN_TRACING_V2=true`).

For work with sensitive code, either:

1. Keep tracing disabled (`LANGCHAIN_TRACING_V2=false` or not set).
2. Use LangSmith's self-hosted option (Enterprise tier).
3. Use Langfuse self-hosted (§18.14).

### Disabling Tracing Per-Request

If tracing is generally enabled but you want to disable it for a specific run:

```python
from langchain_core.runnables.config import RunnableConfig
from langsmith.run_helpers import tracing_context

# Disable tracing for this invocation.
with tracing_context(enabled=False):
    result = await graph.ainvoke(state, config)
```

This is useful for a "private mode" feature where the user explicitly opts out of tracing for sensitive tasks.

### Data Retention

LangSmith's free tier retains traces for 14 days. After that, they are automatically deleted. For longer retention, the paid tier is required.

---

## 18.14 Alternatives to LangSmith

LangSmith is the most polished option for LangChain/LangGraph projects, but alternatives exist.

### Langfuse

[Langfuse](https://langfuse.com) is an open-source LLM observability platform. It is functionally similar to LangSmith: traces, datasets, evaluations, dashboards.

**Advantages over LangSmith**:

- **Self-hosted**. Run Langfuse on your own infrastructure with `docker compose up`. Your traces never leave your servers.
- **Open-source**. Apache 2.0 licence. You can inspect and modify the code.
- **Framework-agnostic**. Works with LangChain, but also raw OpenAI API calls, Anthropic SDK, etc.
- **Free self-hosted tier**. No trace limits when running your own instance.

**Setup for saathi with Langfuse**:

```bash
# Clone and run Langfuse locally.
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d

# In .env:
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

Integration with LangChain:

```python
from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler()

config = {"callbacks": [langfuse_handler]}
result = await graph.ainvoke(state, config=config)
```

### Arize Phoenix

[Arize Phoenix](https://phoenix.arize.com) is an open-source ML observability tool. It focuses on model monitoring and drift detection, with LLM tracing added more recently.

**When to use Phoenix**: When you are also monitoring traditional ML models alongside LLM features. Phoenix's strength is unified ML + LLM observability on a single platform.

**Setup**:

```bash
pip install arize-phoenix openinference-instrumentation-langchain

import phoenix as px
px.launch_app()  # starts local UI at http://localhost:6006

from openinference.instrumentation.langchain import LangChainInstrumentor
LangChainInstrumentor().instrument()
```

No code changes to the LangGraph graph—Phoenix instruments at the LangChain SDK level.

### Weights & Biases Weave

[Weights & Biases Weave](https://wandb.ai/site/weave) extends W&B's ML experiment tracking platform to LLM tracing and evaluation.

**When to use Weave**: When your team already uses W&B for ML experiment tracking. Weave adds LLM observability to the same platform where you track model training runs, hyperparameter sweeps, and model evaluation metrics.

**Integration**:

```python
import weave

weave.init("saathi-langgraph")

@weave.op()
async def run_session_turn(task: str, state: dict) -> dict:
    # ... graph invocation ...
```

### Comparison Summary

| Feature | LangSmith | Langfuse | Phoenix | W&B Weave |
| --------- | ----------- | ---------- | --------- | ----------- |
| LangGraph integration | Native | Via callbacks | Via OpenInference | Via decorator |
| Self-hosted | Enterprise | Free (OSS) | Free (OSS) | No |
| Datasets / Eval | Yes | Yes | Limited | Yes |
| ML model tracking | No | No | Yes | Yes |
| Free tier | 5k traces/mo | Unlimited (self-hosted) | Unlimited (local) | 100GB storage |
| Setup complexity | Minimal | Medium | Low | Low |

**For saathi development**: Start with LangSmith for its native LangGraph integration and polished UI. Switch to Langfuse self-hosted if privacy is a concern or if you need unlimited traces.

---

## 18.15 Structured Logging vs LangSmith Tracing — How They Complement Each Other

Saathi uses two observability mechanisms:

1. **structlog** (or standard Python `logging`): Structured logs written to stderr. Available locally, immediately, without any network call.
2. **LangSmith**: Cloud-hosted traces with rich UI. Available asynchronously, with deep per-call detail.

These are not competitors—they serve different use cases.

### structlog for Fast Local Debugging

During development, you do not want to open a browser every time you add a `print()` statement. structlog gives you immediate, structured output in the terminal:

```bash
2026-07-09 14:30:01 [info     ] saathi starting  model=qwen2.5:14b debug=false
2026-07-09 14:30:01 [debug    ] graph built      nodes=['agent', 'tools'] edges=3
2026-07-09 14:30:02 [debug    ] agent node       input_tokens=220 tool_calls=1
2026-07-09 14:30:02 [debug    ] tool call        tool=read_file path=graph.py
2026-07-09 14:30:02 [debug    ] tool result      tool=read_file chars=1842 ok=true
2026-07-09 14:30:02 [debug    ] agent node       input_tokens=414 output_tokens=153
2026-07-09 14:30:02 [info     ] turn complete    elapsed=2.1s
```

This is the fastest feedback loop. No browser, no network, no latency.

### LangSmith for Deep Analysis

For questions that require seeing the full prompt (not just token counts), understanding multi-session trends, running evaluations, or debugging subtle model behaviour, structlog is insufficient. LangSmith provides:

- The exact prompt as rendered by LangChain (including the full tool definitions JSON)
- Token counts attributed to specific message types (system prompt vs conversation history vs tool results)
- The model's full completion, including tool call JSON
- Trend data across many runs
- Dataset-based evaluation to measure quality changes

### Recommended Workflow

| Situation | Use |
| ----------- | ----- |
| Adding a new feature, quick iteration | structlog (`--debug` flag) |
| "Why did the agent call the wrong tool?" | LangSmith trace detail |
| "How many tokens does my system prompt use?" | LangSmith prompt viewer |
| "Did this change improve response quality?" | LangSmith evaluation |
| "Is there a latency regression in production?" | LangSmith dashboard |
| "What did the model say 3 sessions ago?" | LangSmith trace search |

### Configuring structlog in saathi

```python
# src/saathi/logging_config.py
"""Structured logging configuration for saathi."""

import logging
import sys
from typing import Literal

try:
    import structlog

    def configure_structlog(log_level: str = "INFO") -> None:
        """Configure structlog with a clean, colourised output format."""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stderr,
            level=getattr(logging, log_level.upper(), logging.INFO),
        )

except ImportError:
    # structlog not installed; fall back to standard logging.
    def configure_structlog(log_level: str = "INFO") -> None:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stderr,
            level=getattr(logging, log_level.upper(), logging.INFO),
        )
```

This is called once at CLI startup in `cli.py`:

```python
from saathi.logging_config import configure_structlog

# In main():
configure_structlog(log_level=cfg.log_level)
```

With `SAATHI_LOG_LEVEL=DEBUG` (or `--debug`), structlog outputs every LangChain event to stderr. With `SAATHI_LOG_LEVEL=INFO` (the default), only high-level events are logged.

### The Layered Observability Stack

```folder
┌─────────────────────────────────────────────────────────────────────┐
│  LangSmith (cloud, rich UI)                                         │
│  ├─ Traces for every graph run                                      │
│  ├─ Prompt/completion viewer                                        │
│  ├─ Token aggregation across sessions                               │
│  ├─ Dataset-based evaluation                                        │
│  └─ Trend dashboards                                                │
├─────────────────────────────────────────────────────────────────────┤
│  structlog (local, terminal)                                        │
│  ├─ Per-turn events: node executions, tool calls, token counts      │
│  ├─ Error traces and exception details                              │
│  └─ Debug-level event stream for fast iteration                     │
├─────────────────────────────────────────────────────────────────────┤
│  Rich console (saathi UI)                                           │
│  ├─ Agent responses rendered with Markdown                          │
│  ├─ Tool call notifications (one line per tool)                     │
│  └─ Token usage footer (↳ 1,240 in · 312 out · 1.8s)               │
└─────────────────────────────────────────────────────────────────────┘
```

Each layer serves a different consumer and a different timescale. The Rich console is for the user watching in real-time. structlog is for the developer iterating locally. LangSmith is for the developer doing post-hoc analysis or systematic evaluation.

---

## 18.16 Enabling LangSmith in CI

For a production-grade saathi deployment, enable LangSmith tracing in CI to capture evaluation runs alongside unit tests.

### GitHub Actions Example

```yaml
# .github/workflows/eval.yml
name: LangSmith Evaluation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    services:
      ollama:
        image: ollama/ollama
        ports:
          - 11434:11434

    env:
      SAATHI_MODEL: llama3.2:3b
      SAATHI_OLLAMA_BASE_URL: http://localhost:11434
      LANGCHAIN_TRACING_V2: "true"
      LANGCHAIN_PROJECT: saathi-ci
      LANGCHAIN_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}

    steps:
      - uses: actions/checkout@v4

      - name: Pull model
        run: |
          # Wait for Ollama to start.
          until curl -sf http://localhost:11434/api/tags; do sleep 2; done
          ollama pull llama3.2:3b

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run unit tests
        run: pytest tests/ -v

      - name: Run LangSmith evaluation
        run: python scripts/run_evaluation.py
```

```python
# scripts/run_evaluation.py
"""Run LangSmith evaluation in CI."""

from langsmith.evaluation import evaluate
from saathi.eval import run_saathi, load_evaluators

results = evaluate(
    run_saathi,
    data="saathi-qa",
    evaluators=load_evaluators(),
    experiment_prefix=f"ci-{os.environ.get('GITHUB_SHA', 'local')[:8]}",
    metadata={
        "branch": os.environ.get("GITHUB_REF_NAME", "unknown"),
        "commit": os.environ.get("GITHUB_SHA", "unknown"),
        "model": os.environ.get("SAATHI_MODEL", "unknown"),
    },
)

# Fail CI if average correctness score is below threshold.
avg_correctness = results.to_pandas()["llm_correctness"].mean()
if avg_correctness < 0.7:
    print(f"Evaluation failed: average correctness {avg_correctness:.2f} < 0.70")
    sys.exit(1)

print(f"Evaluation passed: average correctness {avg_correctness:.2f}")
```

This CI workflow runs the evaluation dataset on every push to `main` and fails if quality drops below a threshold. Combined with LangSmith's comparison view (which shows before/after differences for each experiment), it creates a complete quality gate for AI application development.

---

## Summary

LangSmith solves the observability problem for LLM applications that traditional APM tools cannot address. The key points:

- **Setup is two environment variables**: `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY`. No code changes.
- **Every LangGraph run is automatically traced**: nodes, tool calls, LLM calls, token counts, latency—all captured.
- **The LangSmith UI** shows the full trace tree, exact prompts, model completions, and latency breakdowns.
- **Debugging becomes observation**: instead of guessing why the agent misbehaved, open the trace and read the exact prompt it received.
- **Token tracking** enables context window monitoring, model comparison, and cost projection for cloud LLMs.
- **Run metadata and tags** make traces filterable and searchable across sessions and deploys.
- **Datasets and evaluations** enable systematic quality measurement and regression testing.
- **Custom evaluators** measure domain-specific quality with either heuristics or LLM-as-judge.
- **`@traceable`** adds custom Python functions to the trace tree alongside LangChain operations.
- **Privacy**: tracing is opt-in; traces contain prompt content; use Langfuse self-hosted for sensitive workloads.
- **Alternatives**: Langfuse (open-source, self-hosted), Arize Phoenix (unified ML+LLM), W&B Weave (for W&B users).
- **structlog + LangSmith are complementary**: structlog for fast local iteration, LangSmith for deep analysis and evaluation.

To enable LangSmith tracing in saathi, add three lines to your `.env` file and run any session. The traces appear in the LangSmith UI within seconds.
