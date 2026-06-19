# 04 — The ReAct Loop, Built From Scratch

*Part 4 of 5 — From Neuron to Agent*
*Series index: [README.md](./README.md)*

---

## Overview

This document covers `agent.py` — the orchestration loop that connects the model, the tools, and the prompts into a working research agent.

The loop is an implementation of the ReAct pattern (Yao et al., 2022): **Re**ason, **Act**, **Observe**, repeat. This document explains the pattern, maps it to the implementation, covers every failure mode and how it is handled, and shows what LangChain's `AgentExecutor` abstracts when you eventually move to Phase 2.

---

## The ReAct pattern

ReAct is a prompting and orchestration strategy that interleaves reasoning steps with action steps. The model does not answer a question in one pass. It reasons about what to do, acts by calling a tool, observes the result, and reasons again from the updated context.

Original paper: <https://arxiv.org/abs/2210.03629>

General form:

```text
Thought: what do I need to find out?
Action:  call a tool with a specific input
Observe: read the tool output
Thought: what does this tell me? what next?
Action:  call another tool, or stop and answer
```

The loop terminates when the model determines it has enough information to answer, or when a maximum step count is reached.

---

## This project's implementation

The research agent implements a constrained two-step ReAct loop. Rather than letting the model decide when to call tools and when to stop, the structure is fixed: one planner call, N search calls, one synthesizer call. This is deliberate — an open-ended loop is harder to debug and adds failure modes that are not necessary for a research pipeline.

```text
Thought (Call 1):  Planner — break question into sub-questions
Action:            search_web() for each sub-question
Observe:           formatted search results
Thought (Call 2):  Synthesizer — write answer from results
Answer:            return final_answer
```

This is ReAct with a fixed graph rather than a dynamic one. The model reasons twice. The tool is called N times between the two reasoning steps. The loop does not recurse.

---

## Full implementation

### `agent.py`

```python
"""
agent.py — Gemma 4 Research Agent
Orchestrates the full plan → search → synthesize pipeline.

Pipeline:
    1. Planner   : Gemma 4 breaks the user question into 2-3 sub-questions
    2. Search    : ddgs fetches web results for each sub-question
    3. Synthesizer: Gemma 4 reads all results and writes a structured answer

References:
    ReAct paper:    https://arxiv.org/abs/2210.03629
    Gemma 4 card:   https://huggingface.co/google/gemma-4-12B-it
    ddgs library:   https://github.com/deedy5/ddgs
"""

from model import generate_response
from tools import search_web, format_search_results_for_prompt
from prompts import (
    PLANNER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    build_planner_user_prompt,
    build_synthesizer_user_prompt,
    parse_sub_questions,
)


def run_research_agent(model, processor, user_question: str) -> str:
    """
    Runs the full research agent pipeline for a given question.

    Steps:
        1. Planner call  — Gemma 4 produces 2-3 sub-questions
        2. Search loop   — ddgs fetches results for each sub-question
        3. Synthesizer call — Gemma 4 writes a structured answer from results

    Args:
        model: Loaded Gemma 4 model (from model.py).
        processor: Loaded Gemma 4 processor (from model.py).
        user_question (str): The research question from the user.

    Returns:
        str: The final synthesized answer.
    """

    _print_step_header("RESEARCH AGENT STARTING")
    print(f"Question: {user_question}\n")

    # -------------------------------------------------------------------
    # Step 1 — Planner
    # -------------------------------------------------------------------
    _print_step_header("STEP 1: PLANNING")
    print("Asking Gemma 4 to break the question into sub-questions...\n")

    planner_response = generate_response(
        model=model,
        processor=processor,
        user_message=build_planner_user_prompt(user_question),
        system_message=PLANNER_SYSTEM_PROMPT
    )

    print(f"Planner raw output:\n{planner_response}\n")

    sub_questions = parse_sub_questions(planner_response)

    if not sub_questions:
        print("[agent.py] Planner returned no parseable sub-questions. Aborting.")
        return "Agent failed: planner did not produce valid sub-questions."

    print(f"Parsed {len(sub_questions)} sub-questions:")
    for index, sub_question in enumerate(sub_questions, start=1):
        print(f"  {index}. {sub_question}")

    # -------------------------------------------------------------------
    # Step 2 — Search
    # -------------------------------------------------------------------
    _print_step_header("STEP 2: SEARCHING")

    all_formatted_results = []

    for index, sub_question in enumerate(sub_questions, start=1):
        print(f"[{index}/{len(sub_questions)}] Searching: {sub_question}")

        search_results = search_web(sub_question)

        if not search_results:
            print(f"  No results found for sub-question {index}. Skipping.\n")
            continue

        formatted_results = format_search_results_for_prompt(
            query=sub_question,
            results=search_results
        )

        all_formatted_results.append(formatted_results)
        print(f"  Found {len(search_results)} results.\n")

    if not all_formatted_results:
        print("[agent.py] All searches returned empty. Aborting.")
        return "Agent failed: no search results were returned for any sub-question."

    combined_search_results = "\n\n".join(all_formatted_results)

    # -------------------------------------------------------------------
    # Step 3 — Synthesizer
    # -------------------------------------------------------------------
    _print_step_header("STEP 3: SYNTHESIZING")
    print("Asking Gemma 4 to synthesize a final answer from search results...\n")

    final_answer = generate_response(
        model=model,
        processor=processor,
        user_message=build_synthesizer_user_prompt(
            user_question=user_question,
            all_search_results_text=combined_search_results
        ),
        system_message=SYNTHESIZER_SYSTEM_PROMPT
    )

    _print_step_header("FINAL ANSWER")
    print(final_answer)

    return final_answer


def _print_step_header(title: str) -> None:
    """Prints a clearly visible step separator for notebook readability."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")
```

---

## Failure modes and handling

Every agent loop has failure modes. This implementation handles three explicit ones.

### Failure 1 — Planner returns unparseable output

**Cause:** The model adds preamble, explanation, or uses a format other than a numbered list despite the system prompt constraints.

**Detection:**

```python
sub_questions = parse_sub_questions(planner_response)

if not sub_questions:
    print("[agent.py] Planner returned no parseable sub-questions. Aborting.")
    return "Agent failed: planner did not produce valid sub-questions."
```

**Why abort rather than retry:** A retry without changing the prompt produces the same output. Retrying with a modified prompt requires prompt versioning logic that is out of scope for this implementation. Aborting with a clear message is preferable to silently continuing with zero sub-questions, which would pass an empty list to the search step.

---

### Failure 2 — Single search returns empty

**Cause:** ddgs rate-limiting, network error, or a sub-question too narrow to match any results.

**Detection:**

```python
search_results = search_web(sub_question)

if not search_results:
    print(f"  No results found for sub-question {index}. Skipping.\n")
    continue
```

**Why skip rather than abort:** A partial result set is more useful than no result set. If two out of three sub-questions return results, the synthesizer can still produce a useful answer. The synthesizer prompt instructs the model to acknowledge when information is insufficient.

---

### Failure 3 — All searches return empty

**Cause:** Network connectivity failure, ddgs service unavailable, or all sub-questions too narrow.

**Detection:**

```python
if not all_formatted_results:
    print("[agent.py] All searches returned empty. Aborting.")
    return "Agent failed: no search results were returned for any sub-question."
```

**Why abort here:** Calling the synthesizer with no search results produces a response that bypasses the grounding constraint — the model falls back to its training data and hallucination risk rises. Aborting with a clear message is the correct behaviour.

---

## Separation of concerns

`agent.py` receives an already-loaded model and processor. It never calls `load_model_and_processor()` directly. This is intentional.

| File | Responsibility |
| --- | --- |
| `model.py` | Model loading, inference, thinking token stripping |
| `tools.py` | Web search, result formatting |
| `prompts.py` | Prompt templates, output parsing |
| `agent.py` | Orchestration only — calls the above, handles failures |
| `notebook.ipynb` | Environment setup, model loading, entry point |

The model is loaded once in the notebook. `run_research_agent()` is called as many times as needed against the loaded model. Loading a 12B model takes approximately 2 minutes. The orchestration loop takes approximately 30–60 seconds per run depending on model generation time.

---

## What LangChain abstracts (Phase 2 preview)

The raw loop in `agent.py` maps directly to LangChain primitives. Understanding the raw implementation first makes the abstractions legible.

| `agent.py` (raw) | LangChain equivalent |
| --- | --- |
| `generate_response()` | `HuggingFacePipeline` LLM wrapper |
| `search_web()` | `DuckDuckGoSearchRun` tool |
| `run_research_agent()` loop | `AgentExecutor` + ReAct agent |
| `build_planner_user_prompt()` | `ChatPromptTemplate` |
| `parse_sub_questions()` | `StrOutputParser` / custom parser |
| `_print_step_header()` | LangSmith trace (visual, not print) |

LangChain does not change what happens. It changes how much of it you write yourself. The tradeoff is abstraction cost — fewer lines of code, less visibility into the loop, and a dependency on a framework that evolves rapidly.

Phase 2 of this project refactors the raw pipeline to LangChain component by component, running the notebook after each swap to confirm the output is identical.

---

## Verified pipeline output

Input:

```text
user_question = "How do transformer models handle long-range dependencies?"
```

Step 1 — Planner output:

```text
1. What is the role of the self-attention mechanism in capturing dependencies between tokens?
2. What are the limitations of the standard transformer architecture regarding sequence length and memory complexity?
3. What techniques, such as sparse attention or sliding windows, are used to optimize long-range dependency handling?
```

Step 2 — Search:

```text
[1/3] Searching: What is the role of the self-attention mechanism...
  Found 5 results.
[2/3] Searching: What are the limitations of the standard transformer...
  Found 5 results.
[3/3] Searching: What techniques, such as sparse attention...
  Found 5 results.
```

Step 3 — Synthesizer produced a structured answer with introduction, key findings across three categories (self-attention role, computational constraints, optimization techniques), inline citations `[1]` through `[5]`, and a conclusion. Full output in the Kaggle notebook.

---

## References

- ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022): <https://arxiv.org/abs/2210.03629>
- LangChain AgentExecutor: <https://python.langchain.com/docs/modules/agents/>
- LangChain HuggingFacePipeline: <https://python.langchain.com/docs/integrations/llms/huggingface_pipeline>
- ddgs library: <https://github.com/deedy5/ddgs>
- Gemma 4 model card: <https://huggingface.co/google/gemma-4-12B-it>
- Agent source: [../Research_Agent_Gemma4/agent.py](../Research_Agent_Gemma4/agent.py)

---

*[← 03 — Prompt Engineering as Design](./03-prompt-engineering-as-design.md) | [Series Index](./README.md) | [Next: 05 — Dense vs MoE →](./05-dense-vs-moe.md)*
