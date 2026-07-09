"""Conversation history compaction to stay within the context window.

When history grows large, we summarize the older turns into a single message and
keep the most recent turns verbatim. The cut is made at a **user-turn boundary**
so the retained tail is always a valid message sequence — it never begins with an
orphaned ``ToolMessage`` whose ``AIMessage`` (tool call) was summarized away.
"""

from __future__ import annotations

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

_SUMMARY_PREFIX = "Summary of earlier conversation:"
_CHARS_PER_TOKEN = 4

_SUMMARY_INSTRUCTIONS = (
    "You are compacting a coding-assistant conversation to save context window. "
    "Write a concise summary capturing: the user's goals, key decisions, files "
    "read or modified, important findings, and any unresolved threads. Preserve "
    "concrete details a developer would need to continue. Output only the summary."
)


def _text(message: BaseMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content)


def estimate_tokens(messages: list[BaseMessage]) -> int:
    """Rough token estimate (~4 chars per token) across message contents."""
    return sum(len(_text(m)) for m in messages) // _CHARS_PER_TOKEN


def needs_compaction(messages: list[BaseMessage], budget_tokens: int) -> bool:
    return estimate_tokens(messages) > budget_tokens


def split_for_compaction(
    messages: list[BaseMessage], keep_turns: int
) -> tuple[list[BaseMessage], list[BaseMessage]] | None:
    """Split into ``(older, recent)`` at a user-turn boundary.

    Keeps the last ``keep_turns`` user turns (and everything after them) intact.
    Returns ``None`` when there aren't more than ``keep_turns`` turns — i.e. there
    is nothing worth compacting yet.
    """
    human_idxs = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_idxs) <= keep_turns:
        return None
    cut = human_idxs[-keep_turns]
    return messages[:cut], messages[cut:]


async def compact_messages(
    llm: LanguageModelLike,
    messages: list[BaseMessage],
    *,
    keep_turns: int = 3,
) -> list[BaseMessage]:
    """Summarize all but the last ``keep_turns`` turns into one summary message.

    Returns ``[summary, *recent]``, or the original list **unchanged** (same
    object) when there is not enough history to compact.
    """
    split = split_for_compaction(messages, keep_turns)
    if split is None:
        return messages
    older, recent = split

    transcript = "\n".join(
        f"{m.__class__.__name__.replace('Message', '')}: {_text(m)}" for m in older
    )
    response = await llm.ainvoke(
        [
            SystemMessage(content=_SUMMARY_INSTRUCTIONS),
            HumanMessage(content=f"Conversation so far:\n\n{transcript}"),
        ]
    )
    summary_text = _text(response) if isinstance(response, BaseMessage) else str(response)
    summary = SystemMessage(content=f"{_SUMMARY_PREFIX}\n{summary_text}")
    return [summary, *recent]
