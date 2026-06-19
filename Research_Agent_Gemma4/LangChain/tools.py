"""
tools.py — Gemma 4 Research Agent (Phase 2 — LangChain)
Web search tool using LangChain's DuckDuckGoSearchRun.

No API key required. Uses DuckDuckGo via langchain-community.

References:
    DuckDuckGoSearchRun:     https://python.langchain.com/docs/integrations/tools/ddg/
    DuckDuckGoSearchAPIWrapper: https://api.python.langchain.com/en/latest/community/utilities/langchain_community.utilities.duckduckgo_search.DuckDuckGoSearchAPIWrapper.html
"""

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# ---------------------------------------------------------------------------
# Search tool
# ---------------------------------------------------------------------------

def build_search_tool(max_results: int = 3) -> DuckDuckGoSearchRun:
    """
    Builds a LangChain DuckDuckGoSearchRun tool.

    Returns a single concatenated string of search results per query.
    max_results is set to 3 by default to keep synthesizer input size
    manageable within the model's context window.

    Args:
        max_results (int): Number of search results per query. Defaults to 3.

    Returns:
        DuckDuckGoSearchRun: LangChain-compatible search tool.

    References:
        https://python.langchain.com/docs/integrations/tools/ddg/
    """
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)
    search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
    return search_tool


# ---------------------------------------------------------------------------
# Search runner
# ---------------------------------------------------------------------------

def run_searches(sub_questions: list[str], search_tool: DuckDuckGoSearchRun) -> str:
    """
    Runs a web search for each sub-question and returns all results
    concatenated into a single string for the synthesizer.

    Each result block is labeled with its query for context.

    Args:
        sub_questions (list[str]): List of sub-questions from the planner.
        search_tool (DuckDuckGoSearchRun): Initialised search tool.

    Returns:
        str: All search results concatenated. Empty string if all fail.
    """
    all_results = []

    for index, question in enumerate(sub_questions, start=1):
        print(f"[{index}/{len(sub_questions)}] Searching: {question}")

        try:
            result = search_tool.invoke(question)
            if result:
                all_results.append(f'Search results for: "{question}"\n{result}')
                print(f"  Done.\n")
            else:
                print(f"  No results. Skipping.\n")
        except Exception as search_error:
            print(f"  Search failed: {search_error}. Skipping.\n")

    return "\n\n".join(all_results)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    search_tool = build_search_tool()

    print("Search tool:", search_tool.name)
    print("Description:", search_tool.description)

    result = search_tool.invoke("transformer neural network attention mechanism")
    print("\nTest result:")
    print(result)