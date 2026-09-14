"""Build an ``LLMClient`` for a role from environment variables.

Roles (§2, §8):

- ``enactment`` — plays the personas in the sim. Centaur over an OpenAI-compatible endpoint when
  ``REGULARS_ENACTMENT_BASE_URL`` is set; otherwise the scoring model plays both sides.
- ``scoring`` — the hosted frontier pass. Defaults to Claude Opus 5.
- ``cheap`` — the Haiku-class first pass of the two-stage ranker (§8). Defaults to Claude Haiku 4.5.

Environment:

- ``REGULARS_ENACTMENT_MODEL``   model id for enactment (default: Centaur when a base URL is
  set, else the scoring model)
- ``REGULARS_ENACTMENT_BASE_URL`` OpenAI-compatible endpoint (vLLM/TGI) for enactment
- ``REGULARS_ENACTMENT_API_KEY``  key for that endpoint (default ``EMPTY``; vLLM ignores it)
- ``REGULARS_SCORING_MODEL``     default ``claude-opus-5``
- ``REGULARS_CHEAP_MODEL``       default ``claude-haiku-4-5``
- ``ANTHROPIC_API_KEY``          forwarded to the Anthropic client; when unset the SDK falls back
  to its own credential resolution (``ANTHROPIC_AUTH_TOKEN`` / ``ant auth login`` profile)

``llm_from_env``'s optional ``on_call`` is an observability hook — ``(duration_seconds,
success)`` — reported after every ``complete()`` call. This module has no dependency on how (or
whether) a caller records it; see ``regulars.observability.instrumented.llm_call_hook`` for a
metrics-recording implementation, built per role since ``on_call`` here is already role-scoped.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from regulars.core.llm import LLMClient, LLMMessage
from regulars.llm.anthropic_client import AnthropicLLM
from regulars.llm.openai_compat import CENTAUR_MODEL, OpenAICompatibleLLM

Role = Literal["enactment", "scoring", "cheap"]

OnCall = Callable[[float, bool], None]


class _InstrumentedLLM:
    """Wraps an ``LLMClient``, reporting ``(duration_seconds, success)`` to ``on_call``."""

    def __init__(self, inner: LLMClient, on_call: OnCall) -> None:
        self._inner = inner
        self._on_call = on_call
        self.model = inner.model

    async def complete(
        self,
        system: str,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> str:
        start = time.monotonic()
        success = True
        try:
            return await self._inner.complete(
                system, messages, temperature=temperature, max_tokens=max_tokens, seed=seed
            )
        except Exception:
            success = False
            raise
        finally:
            self._on_call(time.monotonic() - start, success)


DEFAULT_SCORING_MODEL = "claude-opus-5"
DEFAULT_CHEAP_MODEL = "claude-haiku-4-5"

ENV_ENACTMENT_MODEL = "REGULARS_ENACTMENT_MODEL"
ENV_ENACTMENT_BASE_URL = "REGULARS_ENACTMENT_BASE_URL"
ENV_ENACTMENT_API_KEY = "REGULARS_ENACTMENT_API_KEY"
ENV_SCORING_MODEL = "REGULARS_SCORING_MODEL"
ENV_CHEAP_MODEL = "REGULARS_CHEAP_MODEL"
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"


def llm_from_env(
    role: Role, env: Mapping[str, str] | None = None, *, on_call: OnCall | None = None
) -> LLMClient:
    """Construct the client for ``role``. ``env`` defaults to ``os.environ``.

    ``on_call``, when given, wraps the result so every ``complete()`` reports its latency and
    outcome; ``None`` (the default) returns the provider client unchanged.
    """
    e = os.environ if env is None else env
    api_key = e.get(ENV_ANTHROPIC_API_KEY) or None
    scoring_model = e.get(ENV_SCORING_MODEL) or DEFAULT_SCORING_MODEL

    client: LLMClient
    if role == "scoring":
        client = AnthropicLLM(scoring_model, api_key=api_key)
    elif role == "cheap":
        client = AnthropicLLM(e.get(ENV_CHEAP_MODEL) or DEFAULT_CHEAP_MODEL, api_key=api_key)
    else:
        base_url = e.get(ENV_ENACTMENT_BASE_URL) or None
        if base_url:
            client = OpenAICompatibleLLM(
                e.get(ENV_ENACTMENT_MODEL) or CENTAUR_MODEL,
                base_url=base_url,
                api_key=e.get(ENV_ENACTMENT_API_KEY) or "EMPTY",
            )
        else:
            client = AnthropicLLM(e.get(ENV_ENACTMENT_MODEL) or scoring_model, api_key=api_key)

    if on_call is not None:
        return _InstrumentedLLM(client, on_call)
    return client
