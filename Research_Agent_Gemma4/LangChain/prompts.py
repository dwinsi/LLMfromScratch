"""
prompts.py — Gemma 4 Research Agent (Phase 2 — LangChain)
Prompt templates using LangChain's ChatPromptTemplate.

Phase 1 used plain string functions (build_planner_user_prompt,
build_synthesizer_user_prompt). Phase 2 replaces these with
ChatPromptTemplate objects that integrate directly into LCEL chains.

References:
    ChatPromptTemplate:  https://python.langchain.com/docs/concepts/prompt_templates/
    apply_chat_template: https://huggingface.co/docs/transformers/main/en/chat_templating
    Gemma 4 model card:  https://huggingface.co/google/gemma-4-12B-it
"""

from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------
# Input variable: {question}
# Output: numbered list of 2-3 sub-questions, nothing else
# ---------------------------------------------------------------------------

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a research planner. Your only job is to break a complex question into 2 or 3 focused sub-questions that can each be answered by a single web search.

Rules you must follow without exception:
- Output ONLY a numbered list. No introduction, no explanation, no closing remarks.
- Each sub-question must be specific and self-contained.
- Each sub-question must be on its own line.
- Do not number beyond 3.
- Do not output anything other than the numbered list.

Example output format:
1. What is the history of quantum computing?
2. What are the latest breakthroughs in quantum computing in 2025?
3. Which companies are leading quantum computing research today?"""
    ),
    (
        "human",
        "Break this research question into 2 or 3 focused sub-questions:\n\n{question}"
    )
])


# ---------------------------------------------------------------------------
# Synthesizer prompt
# ---------------------------------------------------------------------------
# Input variables: {question}, {search_results}
# Output: structured answer grounded in search results
# ---------------------------------------------------------------------------

SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a research synthesizer. You receive a research question and a set of web search results. Your job is to write a clear, well-structured answer based strictly on the search results provided.

Rules you must follow:
- Base your answer only on the search results provided. Do not add information from outside the results.
- Structure your answer with a short introduction, key findings, and a brief conclusion.
- Write in plain, precise language. No filler phrases."""
    ),
    (
        "human",
        "Research question: {question}\n\nSearch results:\n{search_results}\n\nWrite a structured answer based on the search results above."
    )
])


# ---------------------------------------------------------------------------
# Sub-question parser
# ---------------------------------------------------------------------------

def parse_sub_questions(planner_output: str) -> list[str]:
    """
    Parses the planner's numbered list output into a Python list.

    Handles numbering variants: "1.", "1)", "1 "
    Returns empty list if no numbered lines are found.

    Args:
        planner_output (str): Plain string output from the planner chain.

    Returns:
        list[str]: Sub-question strings with numbering stripped.
    """
    sub_questions = []

    for line in planner_output.strip().splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned[0].isdigit():
            stripped = cleaned.lstrip("0123456789").lstrip(".").lstrip(")").strip()
            if stripped:
                sub_questions.append(stripped)

    return sub_questions


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Planner prompt input variables:", PLANNER_PROMPT.input_variables)
    print("Synthesizer prompt input variables:", SYNTHESIZER_PROMPT.input_variables)

    mock_planner_output = (
        "1. What is the attention mechanism in transformer models?\n"
        "2. How do transformers differ from RNNs in handling long sequences?\n"
        "3. What are the limitations of transformers with very long contexts?"
    )

    parsed = parse_sub_questions(mock_planner_output)
    for index, question in enumerate(parsed, start=1):
        print(f"Sub-question {index}: {question}")