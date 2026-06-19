"""
agent.py — Gemma 4 Research Agent (Phase 2 — LangChain)
Orchestrates the full plan → search → synthesize pipeline using LCEL.

Phase 1 used a raw Python loop calling generate_response() directly.
Phase 2 replaces each component with LangChain primitives chained via
the pipe operator (|).

Pipeline:
    planner_chain:    PLANNER_PROMPT | chat_model | StrOutputParser() | RunnableLambda(clean_response)
    search:           run_searches() — DuckDuckGoSearchRun × N sub-questions
    synthesizer_chain: SYNTHESIZER_PROMPT | chat_model | StrOutputParser() | RunnableLambda(clean_response)

Phase 1 → Phase 2 mapping:
    generate_response()          → chat_model.invoke() via ChatHuggingFace
    build_planner_user_prompt()  → PLANNER_PROMPT (ChatPromptTemplate)
    build_synthesizer_user_prompt() → SYNTHESIZER_PROMPT (ChatPromptTemplate)
    _strip_thinking_tokens()     → clean_response() via RunnableLambda
    parse_sub_questions()        → parse_sub_questions() (same function)
    search_web()                 → DuckDuckGoSearchRun.invoke()
    format_search_results_for_prompt() → run_searches() returns concatenated string

References:
    LCEL:              https://python.langchain.com/docs/concepts/lcel/
    RunnableLambda:    https://python.langchain.com/docs/concepts/runnables/
    StrOutputParser:   https://python.langchain.com/docs/concepts/output_parsers/
    LangSmith tracing: https://docs.smith.langchain.com/
    ReAct paper:       https://arxiv.org/abs/2210.03629
"""

import os
import warnings
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.context import tracing_v2_enabled

from model import load_model_and_processor, build_chat_model, clean_response
from tools import build_search_tool, run_searches
from prompts import PLANNER_PROMPT, SYNTHESIZER_PROMPT, parse_sub_questions

load_dotenv()
warnings.filterwarnings("ignore", message="Both `max_new_tokens`")


# ---------------------------------------------------------------------------
# Agent pipeline
# ---------------------------------------------------------------------------

def run_research_agent_lcel(
    chat_model,
    search_tool,
    user_question: str,
    use_langsmith: bool = True
) -> str:
    """
    Runs the full LCEL research agent pipeline.

    Steps:
        1. Planner chain  — Gemma 4 produces 2-3 sub-questions
        2. Search loop    — DuckDuckGoSearchRun fetches results per sub-question
        3. Synthesizer chain — Gemma 4 writes structured answer from results

    Args:
        chat_model: ChatHuggingFace instance (from model.py).
        search_tool: DuckDuckGoSearchRun instance (from tools.py).
        user_question (str): The research question from the user.
        use_langsmith (bool): Whether to send traces to LangSmith. Defaults to True.

    Returns:
        str: The final synthesized answer.

    References:
        LCEL: https://python.langchain.com/docs/concepts/lcel/
        LangSmith: https://docs.smith.langchain.com/
    """
    if use_langsmith:
        with tracing_v2_enabled(
            project_name=os.environ.get("LANGCHAIN_PROJECT", "gemma4-research-agent-phase2")
        ):
            return _run_pipeline(chat_model, search_tool, user_question)
    else:
        return _run_pipeline(chat_model, search_tool, user_question)


def _run_pipeline(chat_model, search_tool, user_question: str) -> str:
    """Internal pipeline execution. Called by run_research_agent_lcel."""

    _print_step_header("RESEARCH AGENT STARTING (Phase 2 — LCEL)")
    print(f"Question: {user_question}\n")

    # -------------------------------------------------------------------
    # Step 1 — Planner chain
    # -------------------------------------------------------------------
    _print_step_header("STEP 1: PLANNING")
    print("Running planner chain...\n")

    planner_chain = (
        PLANNER_PROMPT
        | chat_model
        | StrOutputParser()
        | RunnableLambda(clean_response)
    )

    planner_output = planner_chain.invoke({"question": user_question})
    print(f"Planner output:\n{planner_output}\n")

    sub_questions = parse_sub_questions(planner_output)

    if not sub_questions:
        print("[agent.py] Planner returned no parseable sub-questions. Aborting.")
        return "Agent failed: planner did not produce valid sub-questions."

    print(f"Parsed {len(sub_questions)} sub-questions:")
    for index, question in enumerate(sub_questions, start=1):
        print(f"  {index}. {question}")

    # -------------------------------------------------------------------
    # Step 2 — Search
    # -------------------------------------------------------------------
    _print_step_header("STEP 2: SEARCHING")

    search_results = run_searches(sub_questions, search_tool)

    if not search_results:
        print("[agent.py] All searches returned empty. Aborting.")
        return "Agent failed: no search results returned for any sub-question."

    # -------------------------------------------------------------------
    # Step 3 — Synthesizer chain
    # -------------------------------------------------------------------
    _print_step_header("STEP 3: SYNTHESIZING")
    print("Running synthesizer chain...\n")

    synthesizer_chain = (
        SYNTHESIZER_PROMPT
        | chat_model
        | StrOutputParser()
        | RunnableLambda(clean_response)
    )

    final_answer = synthesizer_chain.invoke({
        "question": user_question,
        "search_results": search_results
    })

    _print_step_header("FINAL ANSWER")
    print(final_answer)

    return final_answer


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_step_header(title: str) -> None:
    """Prints a clearly visible step separator for notebook readability."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Entry point — local execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading model...")
    model, processor = load_model_and_processor()
    print("Memory footprint:", round(model.get_memory_footprint() / 1e9, 2), "GB")

    chat_model = build_chat_model(model, processor)
    search_tool = build_search_tool()

    answer = run_research_agent_lcel(
        chat_model=chat_model,
        search_tool=search_tool,
        user_question="How do transformer models handle long-range dependencies?"
    )