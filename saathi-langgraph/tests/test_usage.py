"""Token-usage extraction used for the per-turn footer."""

from langchain_core.messages import AIMessage

from saathi.cli import _extract_usage


def test_usage_metadata() -> None:
    msg = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    assert _extract_usage(msg) == (10, 5)


def test_ollama_response_metadata_fallback() -> None:
    msg = AIMessage(
        content="hi",
        response_metadata={"prompt_eval_count": 7, "eval_count": 3},
    )
    assert _extract_usage(msg) == (7, 3)


def test_no_metadata_returns_none() -> None:
    assert _extract_usage(AIMessage(content="hi")) is None


def test_none_input_returns_none() -> None:
    assert _extract_usage(None) is None
