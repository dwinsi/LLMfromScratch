"""Async retry with exponential backoff for transient failures.

Deliberately narrow: we retry only *connection-establishment* errors (server
not up yet / briefly unreachable), never read timeouts mid-generation — retrying
a slow response would just duplicate output and time out again.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import httpx

# Connection-establishment failures only (pre-stream, safe to re-run).
RETRYABLE: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    ConnectionError,
)


async def retry_async[T](
    func: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    retryable: tuple[type[BaseException], ...] = RETRYABLE,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Call ``func`` (a zero-arg awaitable factory), retrying transient errors.

    Backoff is exponential: ``base_delay * 2**(n-1)`` capped at ``max_delay``.
    Non-retryable exceptions propagate immediately; the last retryable exception
    is re-raised after the final attempt.
    """
    attempts = max(1, attempts)
    for attempt in range(1, attempts + 1):
        try:
            return await func()
        except retryable as exc:
            if attempt == attempts:
                raise
            delay = min(base_delay * 2 ** (attempt - 1), max_delay)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            await sleep(delay)
    # Unreachable: the loop either returns or raises.
    raise AssertionError("retry_async exhausted without returning or raising")
