"""Async retry with exponential backoff."""

import httpx
import pytest

from saathi.retry import retry_async


class _RecordingSleep:
    """A stand-in for asyncio.sleep that records delays instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


async def test_returns_immediately_on_success() -> None:
    sleep = _RecordingSleep()
    calls = 0

    async def ok() -> str:
        nonlocal calls
        calls += 1
        return "done"

    result = await retry_async(ok, sleep=sleep)
    assert result == "done"
    assert calls == 1
    assert sleep.delays == []  # never slept


async def test_retries_then_succeeds() -> None:
    sleep = _RecordingSleep()
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ConnectError("boom")
        return "ok"

    result = await retry_async(flaky, attempts=3, base_delay=1.0, sleep=sleep)
    assert result == "ok"
    assert calls == 3
    assert sleep.delays == [1.0, 2.0]  # exponential backoff between attempts


async def test_raises_after_exhausting_attempts() -> None:
    sleep = _RecordingSleep()

    async def always_fails() -> str:
        raise httpx.ConnectError("down")

    with pytest.raises(httpx.ConnectError):
        await retry_async(always_fails, attempts=3, sleep=sleep)
    assert len(sleep.delays) == 2  # slept between the 3 attempts


async def test_non_retryable_error_propagates_immediately() -> None:
    sleep = _RecordingSleep()
    calls = 0

    async def bad_request() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("not transient")

    with pytest.raises(ValueError):
        await retry_async(bad_request, attempts=3, sleep=sleep)
    assert calls == 1  # no retries
    assert sleep.delays == []


async def test_delay_is_capped_at_max_delay() -> None:
    sleep = _RecordingSleep()

    async def always_fails() -> str:
        raise httpx.ConnectTimeout("slow")

    with pytest.raises(httpx.ConnectTimeout):
        await retry_async(always_fails, attempts=5, base_delay=1.0, max_delay=3.0, sleep=sleep)
    # 1, 2, 4->capped 3, 8->capped 3
    assert sleep.delays == [1.0, 2.0, 3.0, 3.0]


async def test_on_retry_callback_invoked() -> None:
    sleep = _RecordingSleep()
    events: list[tuple[int, float]] = []

    async def flaky() -> str:
        raise httpx.ConnectError("x")

    def on_retry(attempt: int, exc: BaseException, delay: float) -> None:
        events.append((attempt, delay))

    with pytest.raises(httpx.ConnectError):
        await retry_async(flaky, attempts=3, base_delay=1.0, on_retry=on_retry, sleep=sleep)
    assert events == [(1, 1.0), (2, 2.0)]
