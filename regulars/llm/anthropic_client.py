"""Anthropic-hosted ``LLMClient`` (§2 scoring pass, §8 persona-prefix caching).

The system prompt is the per-persona prefix that is identical across every sim a persona takes
part in, so with ``cache_system=True`` it is sent as a single text block carrying
``cache_control: ephemeral``. The dialogue history that follows varies per turn and is not cached.

Retries: the SDK's own retry loop is disabled (``max_retries=0``) and replaced with an explicit
jittered exponential backoff on rate limits and 5xx/overloaded responses, capped at
``max_retries`` attempts, so the retry policy is visible and testable here.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from anthropic import APIStatusError, AsyncAnthropic, RateLimitError
from anthropic.types import (
    CacheControlEphemeralParam,
    MessageParam,
    TextBlock,
    TextBlockParam,
    Usage,
)

from regulars.core.llm import LLMMessage

# Model families on which the Messages API rejects ``temperature`` with a 400 (sampling
# parameters were removed from Claude Opus 4.7 onward). Haiku 4.5 and the 4.6 family still
# accept it. Prefix match on the model id.
_NO_SAMPLING_MODEL_PREFIXES: tuple[str, ...] = (
    "claude-fable-",
    "claude-mythos-",
    "claude-opus-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
)


def supports_temperature(model: str) -> bool:
    """True when the model still accepts a ``temperature`` parameter."""
    return not model.startswith(_NO_SAMPLING_MODEL_PREFIXES)


@dataclass
class UsageTotals:
    """Cumulative token counts across every call made through one client instance."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    calls: int = 0

    def add(self, usage: Usage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += usage.cache_read_input_tokens or 0
        self.cache_creation_tokens += usage.cache_creation_input_tokens or 0
        self.calls += 1


class AnthropicLLM:
    """``LLMClient`` over the async Anthropic SDK.

    ``seed`` is accepted for protocol compatibility and ignored: the Messages API has no seed
    parameter. ``temperature`` is only forwarded on models that accept it
    (see :func:`supports_temperature`).
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        cache_system: bool = True,
        *,
        max_retries: int = 5,
        base_delay_s: float = 1.0,
        max_delay_s: float = 60.0,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self.model = model
        self.cache_system = cache_system
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        # api_key=None lets the SDK resolve ANTHROPIC_API_KEY / an ``ant auth login`` profile.
        self.client = client or AsyncAnthropic(api_key=api_key, max_retries=0)
        self.usage_totals = UsageTotals()
        # Injectable so tests can assert backoff without sleeping.
        self.sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

    def _system_param(self, system: str) -> str | list[TextBlockParam]:
        if not self.cache_system:
            return system
        block: TextBlockParam = {
            "type": "text",
            "text": system,
            "cache_control": CacheControlEphemeralParam(type="ephemeral"),
        }
        return [block]

    def _backoff(self, attempt: int, err: APIStatusError) -> float:
        retry_after = err.response.headers.get("retry-after")
        if retry_after is not None:
            try:
                return min(float(retry_after), self.max_delay_s)
            except ValueError:
                pass
        ceiling = min(self.base_delay_s * (2.0**attempt), self.max_delay_s)
        return ceiling * random.uniform(0.5, 1.5)

    @staticmethod
    def _retryable(err: APIStatusError) -> bool:
        return isinstance(err, RateLimitError) or err.status_code >= 500

    async def complete(
        self,
        system: str,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> str:
        params: list[MessageParam] = [{"role": m.role, "content": m.content} for m in messages]
        # The SDK's typed surface no longer carries sampling parameters; models that still accept
        # ``temperature`` get it through the request body escape hatch.
        extra_body: dict[str, float] | None = (
            {"temperature": temperature} if supports_temperature(self.model) else None
        )
        attempt = 0
        while True:
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=self._system_param(system),
                    messages=params,
                    extra_body=extra_body,
                )
            except APIStatusError as err:
                if not self._retryable(err) or attempt >= self.max_retries:
                    raise
                await self.sleep(self._backoff(attempt, err))
                attempt += 1
                continue
            self.usage_totals.add(response.usage)
            return "".join(
                block.text for block in response.content if isinstance(block, TextBlock)
            ).strip()
