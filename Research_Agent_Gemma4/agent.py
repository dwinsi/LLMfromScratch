"""
agent.py — Gemma 4 Research Agent
Orchestrates the full plan → search → synthesize pipeline.

Pipeline:
    1. Planner   : Gemma 4 breaks the user question into 2-3 sub-questions
    2. Search    : ddgs fetches web results for each sub-question
    3. Synthesizer: Gemma 4 reads all results and writes a structured answer

This file deliberately contains no model loading — that is model.py's job.
agent.py receives an already-loaded model and processor, keeping concerns separate.

References:
    model.py      : Model loading and generate_response()
    tools.py      : search_web() and format_search_results_for_prompt()
    prompts.py    : Prompt templates and parse_sub_questions()
    Gemma 4 card  : https://huggingface.co/google/gemma-4-12B-it
    ddgs library  : https://github.com/deedy5/ddgs
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


# ---------------------------------------------------------------------------
# Agent pipeline
# ---------------------------------------------------------------------------

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

    # Concatenate all result blocks into one context string for the synthesizer
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_step_header(title: str) -> None:
    """Prints a clearly visible step separator for notebook readability."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Sanity check — run directly to verify the full pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Model and processor must already be loaded before calling this.
    # In the Kaggle notebook, model and processor are loaded in earlier cells.
    # This block is a reminder — it will raise NameError if run without them.
    print("agent.py loaded. Call run_research_agent(model, processor, question) to start.")