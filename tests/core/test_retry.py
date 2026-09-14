"""``regulars.core.retry``: shared jittered backoff used by regulars.llm and regulars.payments."""

from __future__ import annotations

import pytest

from regulars.core.retry import jittered_backoff_delay, retry_async


def test_jittered_backoff_delay_bounds() -> None:
    # attempt 0: ceiling = 1.0 -> [0.5, 1.5]
    assert 0.5 <= jittered_backoff_delay(0, 1.0, 60.0, jitter=lambda a, b: a) <= 1.5
    assert jittered_backoff_delay(0, 1.0, 60.0, jitter=lambda lo, hi: lo) == 0.5
    assert jittered_backoff_delay(0, 1.0, 60.0, jitter=lambda lo, hi: hi) == 1.5
    # ceiling is capped at max_delay_s regardless of how large the exponential term gets
    assert jittered_backoff_delay(10, 1.0, 5.0, jitter=lambda lo, hi: 1.0) == 5.0


async def test_retry_async_retries_then_succeeds() -> None:
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ValueError("transient")
        return "ok"

    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    result = await retry_async(
        flaky,
        is_retryable=lambda exc: isinstance(exc, ValueError),
        max_retries=5,
        sleep=fake_sleep,
    )
    assert result == "ok"
    assert calls == 3
    assert len(slept) == 2


async def test_retry_async_gives_up_after_max_retries() -> None:
    calls = 0

    async def always_fails() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("transient")

    async def no_sleep(_: float) -> None:
        return None

    with pytest.raises(ValueError):
        await retry_async(
            always_fails,
            is_retryable=lambda exc: isinstance(exc, ValueError),
            max_retries=2,
            sleep=no_sleep,
        )
    assert calls == 3  # 1 + 2 retries


async def test_retry_async_does_not_retry_non_retryable_errors() -> None:
    calls = 0

    async def fails_hard() -> str:
        nonlocal calls
        calls += 1
        raise KeyError("not transient")

    async def no_sleep(_: float) -> None:
        return None

    with pytest.raises(KeyError):
        await retry_async(
            fails_hard,
            is_retryable=lambda exc: isinstance(exc, ValueError),
            max_retries=5,
            sleep=no_sleep,
        )
    assert calls == 1
