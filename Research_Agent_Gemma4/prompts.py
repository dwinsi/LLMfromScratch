"""
prompts.py — Gemma 4 Research Agent
Prompt templates for the planner and synthesizer steps of the agent loop.

Design principles:
    - Planner output must be strictly parseable — numbered list, nothing else.
    - Synthesizer output must stay grounded in search results provided.
    - System prompts define role and constraints.
    - User prompts inject the dynamic content (question, results).
    - All templates are plain functions returning strings — no frameworks needed.

References:
    Gemma 4 system prompt support:
        https://huggingface.co/google/gemma-4-12B-it
    Gemma 4 chat template / apply_chat_template:
        https://huggingface.co/docs/transformers/main/en/chat_templating
    Gemma 4 best practices (sampling, thinking mode):
        https://ai.google.dev/gemma/docs/capabilities/text/basic
"""


# ---------------------------------------------------------------------------
# Planner prompts
# ---------------------------------------------------------------------------
# The planner receives a user question and breaks it into 2-3 focused
# sub-questions that can each be answered by a single web search.
# Output must be a strict numbered list — agent.py parses this directly.
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """You are a research planner. Your only job is to break a complex question into 2 or 3 focused sub-questions that can each be answered by a single web search.

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


def build_planner_user_prompt(user_question: str) -> str:
    """
    Builds the user turn prompt for the planner step.

    Args:
        user_question (str): The original research question from the user.

    Returns:
        str: Formatted user prompt to pass to generate_response().
    """
    return f"Break this research question into 2 or 3 focused sub-questions:\n\n{user_question}"


# ---------------------------------------------------------------------------
# Synthesizer prompts
# ---------------------------------------------------------------------------
# The synthesizer receives the original question plus all search results
# concatenated together, and writes a structured final answer.
# It must stay grounded — no hallucination beyond what results provide.
# ---------------------------------------------------------------------------

SYNTHESIZER_SYSTEM_PROMPT = """You are a research synthesizer. You receive a research question and a set of web search results. Your job is to write a clear, well-structured answer based strictly on the search results provided.

Rules you must follow:
- Base your answer only on the search results provided. Do not add information from outside the results.
- Structure your answer with a short introduction, key findings, and a brief conclusion.
- When referencing a specific result, cite it by its result number, for example: [1], [2].
- If the search results do not contain enough information to answer the question, say so clearly.
- Write in plain, precise language. No filler phrases."""


def build_synthesizer_user_prompt(
    user_question: str,
    all_search_results_text: str
) -> str:
    """
    Builds the user turn prompt for the synthesizer step.

    Args:
        user_question (str): The original research question from the user.
        all_search_results_text (str): All formatted search results concatenated
            together — output of format_search_results_for_prompt() calls joined
            by newlines.

    Returns:
        str: Formatted user prompt to pass to generate_response().
    """
    return (
        f"Research question: {user_question}\n\n"
        f"Search results:\n{all_search_results_text}\n\n"
        f"Write a structured answer to the research question based on the search results above."
    )


# ---------------------------------------------------------------------------
# Prompt parsing utilities
# ---------------------------------------------------------------------------

def parse_sub_questions(planner_response: str) -> list[str]:
    """
    Parses the planner's numbered list response into a Python list of
    sub-question strings.

    Handles both clean output and cases where the model adds extra whitespace
    or minor formatting variations.

    Example input:
        "1. What is X?\n2. How does Y work?\n3. Why is Z important?"

    Example output:
        ["What is X?", "How does Y work?", "Why is Z important?"]

    Args:
        planner_response (str): Raw text output from the planner model call.

    Returns:
        list[str]: List of sub-question strings with numbering stripped.
                   Returns empty list if parsing finds nothing.
    """
    sub_questions = []

    for line in planner_response.strip().splitlines():
        cleaned_line = line.strip()

        # Skip empty lines
        if not cleaned_line:
            continue

        # Strip leading numbering patterns: "1.", "1)", "1 "
        if cleaned_line[0].isdigit():
            # Remove the digit, and any following ".", ")", or " "
            stripped = cleaned_line.lstrip("0123456789").lstrip(".").lstrip(")").strip()
            if stripped:
                sub_questions.append(stripped)

    return sub_questions


# ---------------------------------------------------------------------------
# Sanity check — run directly to verify prompt structure
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_question = "How do transformer models handle long-range dependencies?"

    print("=== PLANNER SYSTEM PROMPT ===")
    print(PLANNER_SYSTEM_PROMPT)

    print("\n=== PLANNER USER PROMPT ===")
    print(build_planner_user_prompt(test_question))

    print("\n=== SYNTHESIZER SYSTEM PROMPT ===")
    print(SYNTHESIZER_SYSTEM_PROMPT)

    print("\n=== PARSE TEST ===")
    mock_planner_output = (
        "1. What is the attention mechanism in transformer models?\n"
        "2. How do transformers differ from RNNs in handling long sequences?\n"
        "3. What are the limitations of transformers with very long contexts?"
    )
    parsed = parse_sub_questions(mock_planner_output)
    for index, question in enumerate(parsed, start=1):
        print(f"  Sub-question {index}: {question}")