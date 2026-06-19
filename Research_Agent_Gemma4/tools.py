"""
tools.py — Gemma 4 Research Agent
Web search tool used by the agent during the plan → search → synthesize loop.

No API key required. Uses the ddgs metasearch library which aggregates
results from DuckDuckGo and other search engines for free.

References:
    ddgs library (PyPI):   https://pypi.org/project/ddgs/
    ddgs GitHub:           https://github.com/deedy5/ddgs
    DDGS.text() API:       https://github.com/deedy5/ddgs/blob/main/README.md
    DDGS.extract() API:    https://github.com/deedy5/ddgs/blob/main/skills/ddgs/SKILL.md
"""

import json
import pathlib
from ddgs import DDGS

_cfg = json.loads((pathlib.Path(__file__).parent / "config.json").read_text())
MAX_SEARCH_RESULTS = _cfg["tools"]["max_search_results"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

def _build_search_result(title: str, url: str, snippet: str) -> dict:
    """
    Returns a consistently structured search result dict.
    All agent code reads these three keys — title, url, snippet.
    """
    return {
        "title": title,
        "url": url,
        "snippet": snippet
    }


# ---------------------------------------------------------------------------
# Search tool
# ---------------------------------------------------------------------------

def search_web(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict]:
    """
    Searches the web for a query and returns a list of results.

    Each result is a dict with keys:
        title   (str): Page title
        url     (str): Page URL
        snippet (str): Short text excerpt from the page

    Args:
        query (str): The search query string.
        max_results (int): Maximum number of results to return. Defaults to 5.

    Returns:
        list[dict]: List of search result dicts. Returns empty list on failure.

    References:
        https://github.com/deedy5/ddgs/blob/main/README.md
    """
    try:
        raw_results = DDGS().text(
            query,
            region="us-en",
            safesearch="moderate",
            max_results=max_results
        )

        search_results = [
            _build_search_result(
                title=result.get("title", ""),
                url=result.get("href", ""),
                snippet=result.get("body", "")
            )
            for result in raw_results
        ]

        return search_results

    except Exception as search_error:
        print(f"[tools.py] search_web failed for query '{query}': {search_error}")
        return []


def format_search_results_for_prompt(query: str, results: list[dict]) -> str:
    """
    Formats a list of search results into a clean string block suitable
    for injection into a prompt as context.

    Example output:
        Search results for: "quantum computing breakthroughs"

        [1] Title: IBM unveils 1000-qubit processor
            URL: https://example.com/article
            Snippet: IBM announced a new quantum chip...

    Args:
        query (str): The original search query (used as a header).
        results (list[dict]): List of search result dicts from search_web().

    Returns:
        str: Formatted string block ready for prompt injection.
    """
    if not results:
        return f'No results found for: "{query}"'

    formatted_lines = [f'Search results for: "{query}"\n']

    for index, result in enumerate(results, start=1):
        formatted_lines.append(
            f"[{index}] Title: {result['title']}\n"
            f"    URL: {result['url']}\n"
            f"    Snippet: {result['snippet']}\n"
        )

    return "\n".join(formatted_lines)


# ---------------------------------------------------------------------------
# Sanity check — run directly to verify setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_query = "transformer neural network architecture explained"
    print(f"Searching for: {test_query}\n")

    results = search_web(test_query)

    if results:
        print(format_search_results_for_prompt(test_query, results))
    else:
        print("No results returned — check internet connectivity.")