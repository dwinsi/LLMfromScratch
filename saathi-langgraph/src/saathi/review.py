"""Multi-reviewer code review over the working git diff.

A code-orchestrated workflow (not an agent turn): several specialist reviewers
each analyze the diff concurrently and return structured findings with a
confidence score. Findings below a confidence threshold are dropped to cut noise,
and the rest are ranked by severity then confidence.
"""

from __future__ import annotations

import asyncio
import json

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, field_validator
from rich.panel import Panel

from saathi.logging_config import get_logger
from saathi.tools.git import _git
from saathi.ui.display import console

log = get_logger()

_MAX_DIFF_CHARS = 24_000
_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}

# Specialist reviewers: name -> focus instructions.
DEFAULT_REVIEWERS: dict[str, str] = {
    "bugs": (
        "You hunt correctness bugs: logic errors, off-by-one mistakes, wrong "
        "conditions, unhandled None/empty cases, incorrect return values, and "
        "broken control flow."
    ),
    "error-handling": (
        "You hunt error-handling gaps: swallowed or overly broad exceptions, "
        "silent failures, missing validation, and resources that aren't closed."
    ),
    "design": (
        "You review type and interface design: weak or wrong type hints, "
        "inconsistent signatures, leaky abstractions, and needless complexity."
    ),
    "security": (
        "You review security: injection, unsafe shell/eval, path traversal, "
        "hardcoded secrets, and missing input sanitization."
    ),
}


class Finding(BaseModel):
    reviewer: str = ""
    severity: str = "medium"
    confidence: int = 50
    file: str = ""
    line: int | None = None
    issue: str = ""
    suggestion: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: object) -> int:
        try:
            n = int(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 50
        return max(0, min(100, n))

    @field_validator("severity", mode="before")
    @classmethod
    def _normalize_severity(cls, v: object) -> str:
        s = str(v).lower().strip()
        return s if s in _SEVERITY_RANK else "medium"

    @field_validator("line", mode="before")
    @classmethod
    def _coerce_line(cls, v: object) -> int | None:
        if v in (None, "", "null"):
            return None
        try:
            return int(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None


def _extract_json(text: str) -> dict | list | None:
    """Best-effort JSON extraction from an LLM response (tolerates prose/fences)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def parse_findings(text: str, reviewer: str) -> list[Finding]:
    """Parse a reviewer's JSON response into Findings, skipping malformed items."""
    data = _extract_json(text)
    if isinstance(data, dict):
        items = data.get("findings", [])
    elif isinstance(data, list):
        items = data
    else:
        items = []

    findings: list[Finding] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            findings.append(
                Finding(
                    reviewer=reviewer,
                    severity=item.get("severity", "medium"),
                    confidence=item.get("confidence", 50),
                    file=str(item.get("file", "")),
                    line=item.get("line"),
                    issue=str(item.get("issue", "")),
                    suggestion=str(item.get("suggestion", "")),
                )
            )
        except Exception:  # noqa: BLE001 — skip anything that won't validate
            continue
    return findings


def _text(message: BaseMessage | str) -> str:
    if isinstance(message, str):
        return message
    content = message.content
    return content if isinstance(content, str) else str(content)


async def review_one(
    llm: LanguageModelLike, reviewer: str, instructions: str, diff: str
) -> list[Finding]:
    """Run a single specialist reviewer over the diff."""
    system = SystemMessage(
        content=(
            f"You are the '{reviewer}' code reviewer. {instructions}\n\n"
            "Report ONLY issues you are genuinely confident about — no style nits, "
            "no speculation. Respond with a JSON object of this exact shape:\n"
            '{"findings": [{"severity": "high|medium|low", "confidence": 0-100, '
            '"file": "path", "line": <number or null>, "issue": "what is wrong", '
            '"suggestion": "how to fix"}]}\n'
            'If you find nothing, respond with {"findings": []}. Output only JSON.'
        )
    )
    human = HumanMessage(content=f"Review this diff:\n\n{diff}")
    try:
        response = await llm.ainvoke([system, human])
    except Exception as exc:  # noqa: BLE001 — a failed reviewer must not abort the review
        log.warning("reviewer_failed", reviewer=reviewer, error=str(exc))
        return []
    return parse_findings(_text(response), reviewer)


async def run_review(
    llm: LanguageModelLike,
    diff: str,
    *,
    reviewers: dict[str, str] | None = None,
    min_confidence: int = 70,
) -> list[Finding]:
    """Run all reviewers concurrently; return findings >= min_confidence, ranked."""
    reviewers = reviewers or DEFAULT_REVIEWERS
    results = await asyncio.gather(
        *(review_one(llm, name, instr, diff) for name, instr in reviewers.items())
    )
    findings = [f for group in results for f in group if f.confidence >= min_confidence]
    findings.sort(key=lambda f: (_SEVERITY_RANK.get(f.severity, 1), -f.confidence))
    return findings


def get_working_diff() -> str:
    """Return the uncommitted diff vs HEAD (falling back to the unstaged diff)."""
    diff = _git("diff", "HEAD")
    if diff.startswith("Error") or "not a git repository" in diff.lower():
        return ""
    if diff == "(no output)":
        diff = _git("diff")
        if diff == "(no output)" or diff.startswith("Error"):
            return ""
    if len(diff) > _MAX_DIFF_CHARS:
        diff = diff[:_MAX_DIFF_CHARS] + "\n… (diff truncated for review)"
    return diff


_SEVERITY_COLOR = {"high": "red", "medium": "yellow", "low": "blue"}


def render_review(findings: list[Finding], min_confidence: int) -> None:
    """Print review findings as severity-colored panels, most severe first."""
    if not findings:
        console.print(
            f"[green]✓ No findings at or above {min_confidence}% confidence.[/green]"
        )
        return
    console.print(f"\n[bold]Code review — {len(findings)} finding(s)[/bold]\n")
    for f in findings:
        color = _SEVERITY_COLOR.get(f.severity, "yellow")
        loc = f.file + (f":{f.line}" if f.line else "")
        title = (
            f"[{color}]{f.severity.upper()}[/{color}] "
            f"[dim]{f.confidence}%[/dim] [cyan]{loc}[/cyan] [dim]({f.reviewer})[/dim]"
        )
        body = f.issue
        if f.suggestion:
            body += f"\n[dim]→ {f.suggestion}[/dim]"
        console.print(Panel(body, title=title, title_align="left", border_style=color))
