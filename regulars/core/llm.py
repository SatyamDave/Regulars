"""LLM client protocol. Providers live in regulars.llm; tests use FakeLLM.

Every call site takes an ``LLMClient`` so that sim, persona extraction and feature classifiers
run offline in tests and can be pointed at Centaur (OpenAI-compatible endpoint) or a hosted
frontier model without code changes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class LLMMessage:
    role: Role
    content: str


@runtime_checkable
class LLMClient(Protocol):
    """Minimal completion interface. ``seed`` is best-effort; not every provider honors it."""

    model: str

    async def complete(
        self,
        system: str,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> str: ...


@dataclass
class FakeLLM:
    """Deterministic stand-in. Returns scripted replies in order, or calls ``responder``."""

    model: str = "fake"
    replies: list[str] = field(default_factory=list)
    responder: Callable[[str, Sequence[LLMMessage]], str] | None = None
    calls: list[tuple[str, tuple[LLMMessage, ...]]] = field(default_factory=list)

    async def complete(
        self,
        system: str,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> str:
        self.calls.append((system, tuple(messages)))
        if self.responder is not None:
            return self.responder(system, messages)
        if not self.replies:
            raise RuntimeError("FakeLLM has no scripted replies left")
        return self.replies.pop(0)
