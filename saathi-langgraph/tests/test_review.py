"""Multi-reviewer code review: parsing, validation, aggregation, filtering."""

import json

from langchain_core.messages import AIMessage

from saathi.review import (
    Finding,
    _extract_json,
    parse_findings,
    run_review,
)


class FakeLLM:
    """Returns a fixed response for every reviewer call."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    async def ainvoke(self, messages, config=None):  # noqa: ANN001
        self.calls += 1
        return AIMessage(content=self.response)


class RaisingLLM:
    async def ainvoke(self, messages, config=None):  # noqa: ANN001
        raise RuntimeError("model unavailable")


def _resp(findings: list[dict]) -> str:
    return json.dumps({"findings": findings})


# ── Finding validation ─────────────────────────────────────────────────────────
def test_confidence_is_clamped() -> None:
    assert Finding(confidence=150).confidence == 100
    assert Finding(confidence=-5).confidence == 0
    assert Finding(confidence="nonsense").confidence == 50


def test_severity_is_normalized() -> None:
    assert Finding(severity="High").severity == "high"
    assert Finding(severity="CRITICAL").severity == "medium"  # unknown -> medium


def test_line_is_coerced() -> None:
    assert Finding(line="12").line == 12
    assert Finding(line="null").line is None
    assert Finding(line="abc").line is None


# ── JSON extraction ─────────────────────────────────────────────────────────────
def test_extract_json_plain_and_fenced() -> None:
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('prose before {"a": 1} prose after') == {"a": 1}
    assert _extract_json("no json here") is None


# ── parse_findings ──────────────────────────────────────────────────────────────
def test_parse_object_and_array_forms() -> None:
    obj = parse_findings('{"findings": [{"issue": "x", "confidence": 80}]}', "r")
    assert len(obj) == 1 and obj[0].reviewer == "r" and obj[0].confidence == 80
    arr = parse_findings('[{"issue": "y"}]', "r")
    assert len(arr) == 1 and arr[0].confidence == 50  # default


def test_parse_skips_non_dict_and_garbage() -> None:
    assert parse_findings('{"findings": ["oops", {"issue": "ok"}]}', "r") == [
        Finding(reviewer="r", issue="ok")
    ]
    assert parse_findings("totally not json", "r") == []


# ── run_review aggregation / filtering / ranking ────────────────────────────────
async def test_run_review_filters_below_confidence() -> None:
    llm = FakeLLM(
        _resp(
            [
                {"severity": "high", "confidence": 90, "file": "a.py", "issue": "bug"},
                {"severity": "low", "confidence": 40, "file": "b.py", "issue": "nit"},
            ]
        )
    )
    findings = await run_review(llm, "diff", reviewers={"solo": "look"}, min_confidence=70)
    assert len(findings) == 1
    assert findings[0].confidence == 90
    assert findings[0].reviewer == "solo"


async def test_run_review_sorts_by_severity_then_confidence() -> None:
    llm = FakeLLM(
        _resp(
            [
                {"severity": "medium", "confidence": 95, "issue": "m"},
                {"severity": "high", "confidence": 75, "issue": "h"},
            ]
        )
    )
    findings = await run_review(llm, "d", reviewers={"solo": "x"}, min_confidence=70)
    assert [f.severity for f in findings] == ["high", "medium"]


async def test_run_review_aggregates_across_reviewers() -> None:
    llm = FakeLLM(_resp([{"severity": "high", "confidence": 90, "issue": "i"}]))
    findings = await run_review(
        llm, "d", reviewers={"r1": "a", "r2": "b", "r3": "c"}, min_confidence=70
    )
    assert len(findings) == 3
    assert {f.reviewer for f in findings} == {"r1", "r2", "r3"}
    assert llm.calls == 3


async def test_failed_reviewer_yields_no_findings() -> None:
    findings = await run_review(RaisingLLM(), "d", reviewers={"solo": "x"})
    assert findings == []
