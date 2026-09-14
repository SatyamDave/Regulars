"""Jittered exponential backoff shared by outbound-call retry loops.

``regulars.llm.anthropic_client.AnthropicLLM`` originated this jitter formula for the Anthropic
SDK's rate-limit/5xx retries; ``regulars.payments.stripe_deposits.StripeDeposits`` reuses it for
transient Stripe errors. Both keep their own retry loop (the two SDKs raise different exception
types and Anthropic additionally honors a ``Retry-After`` header), but share the pure backoff
computation and a small generic loop for the common case of "retry while ``is_retryable``".
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable


def jittered_backoff_delay(
    attempt: int,
    base_delay_s: float,
    max_delay_s: float,
    *,
    jitter: Callable[[float, float], float] = random.uniform,
) -> float:
    """Exponential backoff ceiling for a zero-indexed ``attempt``, widened by uniform jitter.

    ``ceiling = min(base_delay_s * 2**attempt, max_delay_s)``; the result is
    ``ceiling * jitter(0.5, 1.5)``.
    """
    ceiling = min(base_delay_s * (2.0**attempt), max_delay_s)
    return ceiling * jitter(0.5, 1.5)


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    is_retryable: Callable[[Exception], bool],
    max_retries: int,
    base_delay_s: float = 1.0,
    max_delay_s: float = 60.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run ``operation``, retrying with :func:`jittered_backoff_delay` while ``is_retryable``.

    Gives up (re-raising the last exception) once ``attempt`` reaches ``max_retries``, or
    immediately when ``is_retryable`` returns ``False``.
    """
    attempt = 0
    while True:
        try:
            return await operation()
        except Exception as exc:
            if not is_retryable(exc) or attempt >= max_retries:
                raise
            await sleep(jittered_backoff_delay(attempt, base_delay_s, max_delay_s))
            attempt += 1
