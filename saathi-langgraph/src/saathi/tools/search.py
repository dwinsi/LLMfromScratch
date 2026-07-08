"""Text search and web search tools."""

import re
from pathlib import Path

from langchain_core.tools import tool

from saathi.config import settings

_MAX_FILE_MATCHES = 200


@tool
def search_in_file(path: str, pattern: str) -> str:
    """Search for a regex pattern inside a single file. Returns matching lines with line numbers."""
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: file not found: {path}"
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        matches: list[str] = []
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                matches.append(f"{i}: {line}")
        if not matches:
            return f"No matches for '{pattern}' in {path}"
        return "\n".join(matches[:_MAX_FILE_MATCHES])
    except re.error as e:
        return f"Invalid pattern: {e}"
    except OSError as e:
        return f"Error reading {path}: {e}"


@tool
def search_across_files(
    directory: str,
    pattern: str,
    file_glob: str = "**/*",
) -> str:
    """Recursively search for a regex pattern across all files in a directory."""
    try:
        base = Path(directory)
        if not base.exists():
            return f"Error: directory not found: {directory}"

        regex = re.compile(pattern)
        results: list[str] = []

        for filepath in sorted(base.glob(file_glob)):
            if not filepath.is_file():
                continue
            if filepath.stat().st_size > 500_000:
                continue
            try:
                for i, line in enumerate(
                    filepath.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
                    if regex.search(line):
                        rel = filepath.relative_to(base)
                        results.append(f"{rel}:{i}: {line.strip()}")
                        if len(results) >= _MAX_FILE_MATCHES:
                            results.append(f"... (capped at {_MAX_FILE_MATCHES} matches)")
                            return "\n".join(results)
            except OSError:
                continue

        return "\n".join(results) if results else f"No matches for '{pattern}' in {directory}"
    except re.error as e:
        return f"Invalid pattern: {e}"


@tool
def search_web(query: str) -> str:
    """Search the web for current information using DuckDuckGo (or Brave if configured)."""
    if settings.brave_api_key:
        return _brave_search(query, settings.brave_api_key)
    return _ddg_search(query)


def _ddg_search(query: str) -> str:
    try:
        from duckduckgo_search import DDGS

        results = list(DDGS().text(query, max_results=5))
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(f"**{r.get('title', '')}**\n{r.get('href', '')}\n{r.get('body', '')}\n")
        return "\n---\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


def _brave_search(query: str, api_key: str) -> str:
    try:
        import httpx

        response = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": 5},
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("web", {}).get("results", [])
        lines = []
        for r in results:
            lines.append(
                f"**{r.get('title', '')}**\n{r.get('url', '')}\n{r.get('description', '')}\n"
            )
        return "\n---\n".join(lines) if lines else "No results found."
    except Exception as e:
        return f"Brave search error: {e}"
