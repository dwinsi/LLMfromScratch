"""Git tools — first-class @tool wrappers around common git commands.

These are invoked by the *model* during an agent turn, not by the CLI directly.
Commits happen only when the agent explicitly calls git_commit (usually via /commit).
"""

import subprocess

from langchain_core.tools import tool

_TIMEOUT = 30


def _git(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return "Error: git is not installed or not on PATH."
    except subprocess.TimeoutExpired:
        return f"Error: git {args[0]} timed out after {_TIMEOUT}s."

    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        if "not a git repository" in err.lower():
            return "Error: not a git repository."
        return f"git {args[0]} failed:\n{err}"
    return proc.stdout.strip() or "(no output)"


@tool
def git_status() -> str:
    """Show the working tree status (staged, unstaged, and untracked files)."""
    return _git("status", "--short", "--branch")


@tool
def git_diff(path: str = "") -> str:
    """Show unstaged changes. Optionally limit to a single file or directory path."""
    args = ["diff"]
    if path:
        args.append(path)
    out = _git(*args)
    if len(out) > 20_000:
        return out[:20_000] + "\n… (diff truncated at 20k chars)"
    return out


@tool
def git_diff_staged() -> str:
    """Show staged changes that would be included in the next commit."""
    out = _git("diff", "--cached")
    if len(out) > 20_000:
        return out[:20_000] + "\n… (diff truncated at 20k chars)"
    return out


@tool
def git_log(n: int = 10) -> str:
    """Show the most recent commits (default 10) in a compact one-line format."""
    return _git("log", f"-{max(1, min(n, 50))}", "--oneline", "--decorate")


@tool
def git_commit(message: str, add_all: bool = False) -> str:
    """
    Create a git commit with the given message.
    Set add_all=True to stage all tracked changes first (git add -A).
    Only commit when the user has asked for it.
    """
    if add_all:
        staged = _git("add", "-A")
        if staged.startswith("Error") or "failed" in staged:
            return staged
    return _git("commit", "-m", message)
