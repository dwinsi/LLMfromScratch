"""History compaction: token estimation, turn-boundary splitting, summarizing."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from saathi.compaction import (
    _SUMMARY_PREFIX,
    compact_messages,
    estimate_tokens,
    needs_compaction,
    split_for_compaction,
)


class FakeLLM:
    """Minimal async LLM stub that returns a fixed summary."""

    def __init__(self, summary: str = "THE SUMMARY") -> None:
        self.summary = summary
        self.calls: list[list[BaseMessage]] = []

    async def ainvoke(self, messages, config=None):  # noqa: ANN001
        self.calls.append(messages)
        return AIMessage(content=self.summary)


def _conversation() -> list[BaseMessage]:
    """4 user turns; turns 1 and 3 include a tool-call / tool-result pair."""
    return [
        HumanMessage(content="turn 1"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "t1"}]),
        ToolMessage(content="file contents", tool_call_id="t1", name="read_file"),
        AIMessage(content="answer 1"),
        HumanMessage(content="turn 2"),
        AIMessage(content="answer 2"),
        HumanMessage(content="turn 3"),
        AIMessage(content="", tool_calls=[{"name": "run_bash", "args": {}, "id": "t3"}]),
        ToolMessage(content="cmd output", tool_call_id="t3", name="run_bash"),
        AIMessage(content="answer 3"),
        HumanMessage(content="turn 4"),
        AIMessage(content="answer 4"),
    ]


def test_estimate_tokens() -> None:
    msgs = [HumanMessage(content="a" * 40)]  # 40 chars / 4 = 10
    assert estimate_tokens(msgs) == 10


def test_needs_compaction() -> None:
    msgs = [HumanMessage(content="x" * 400)]  # ~100 tokens
    assert needs_compaction(msgs, budget_tokens=50) is True
    assert needs_compaction(msgs, budget_tokens=500) is False


def test_split_returns_none_when_too_few_turns() -> None:
    msgs = _conversation()  # 4 turns
    assert split_for_compaction(msgs, keep_turns=4) is None
    assert split_for_compaction(msgs, keep_turns=5) is None


def test_split_cuts_at_turn_boundary() -> None:
    msgs = _conversation()
    split = split_for_compaction(msgs, keep_turns=2)
    assert split is not None
    older, recent = split
    # recent keeps the last 2 turns and must start with a HumanMessage
    assert isinstance(recent[0], HumanMessage)
    assert recent[0].content == "turn 3"
    assert older[-1].content == "answer 2"


async def test_compact_returns_unchanged_when_too_few_turns() -> None:
    msgs = _conversation()
    llm = FakeLLM()
    result = await compact_messages(llm, msgs, keep_turns=4)
    assert result is msgs  # same object, no LLM call
    assert llm.calls == []


async def test_compact_summarizes_and_keeps_recent() -> None:
    msgs = _conversation()
    llm = FakeLLM(summary="condensed history")
    result = await compact_messages(llm, msgs, keep_turns=2)

    # summary first, as a SystemMessage carrying the model's text
    assert isinstance(result[0], SystemMessage)
    assert result[0].content.startswith(_SUMMARY_PREFIX)
    assert "condensed history" in result[0].content

    # the retained tail is exactly the last 2 turns, starting on a HumanMessage
    assert isinstance(result[1], HumanMessage)
    assert result[1].content == "turn 3"
    assert [m.content for m in result[-2:]] == ["turn 4", "answer 4"]

    # no orphaned ToolMessage leads the retained tail
    assert not isinstance(result[1], ToolMessage)
    assert llm.calls, "summarizer should have been invoked once"


async def test_compact_shrinks_token_estimate() -> None:
    # Long older turns, short summary -> fewer tokens after compaction.
    msgs = [
        HumanMessage(content="x" * 400),
        AIMessage(content="y" * 400),
        HumanMessage(content="turn 2 " * 10),
        AIMessage(content="answer 2"),
        HumanMessage(content="turn 3"),
        AIMessage(content="answer 3"),
        HumanMessage(content="turn 4"),
        AIMessage(content="answer 4"),
    ]
    before = estimate_tokens(msgs)
    result = await compact_messages(FakeLLM(summary="short"), msgs, keep_turns=2)
    assert estimate_tokens(result) < before
