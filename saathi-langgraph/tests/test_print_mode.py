"""Non-interactive --print / scripting mode."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from saathi.cli import (
    _collect_tool_calls,
    _collect_usage,
    _final_text,
    _print_mode,
)


def test_final_text_picks_last_assistant_message() -> None:
    messages = [
        HumanMessage(content="question"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
        ToolMessage(content="tool result", tool_call_id="1", name="t"),
        AIMessage(content="the final answer"),
    ]
    assert _final_text(messages) == "the final answer"


def test_final_text_empty_when_no_assistant_text() -> None:
    assert _final_text([HumanMessage(content="q")]) == ""


def test_collect_tool_calls() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "read_file", "args": {"path": "a.py"}, "id": "1"},
                {"name": "git_status", "args": {}, "id": "2"},
            ],
        ),
    ]
    assert _collect_tool_calls(messages) == [
        {"name": "read_file", "args": {"path": "a.py"}},
        {"name": "git_status", "args": {}},
    ]


def test_collect_usage_sums_across_messages() -> None:
    messages = [
        AIMessage(
            content="a",
            usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        ),
        AIMessage(
            content="b",
            usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
        ),
    ]
    assert _collect_usage(messages) == {"input_tokens": 8, "output_tokens": 3}


async def test_print_mode_rejects_bad_output_format() -> None:
    # Returns the usage-error code before building the graph — offline-safe.
    code = await _print_mode("gemma4:12b", [], "hi", "yaml")
    assert code == 2
