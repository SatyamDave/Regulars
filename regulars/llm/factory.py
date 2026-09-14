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
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal

from regulars.core.llm import LLMClient
from regulars.llm.anthropic_client import AnthropicLLM
from regulars.llm.openai_compat import CENTAUR_MODEL, OpenAICompatibleLLM

Role = Literal["enactment", "scoring", "cheap"]

DEFAULT_SCORING_MODEL = "claude-opus-5"
DEFAULT_CHEAP_MODEL = "claude-haiku-4-5"

ENV_ENACTMENT_MODEL = "REGULARS_ENACTMENT_MODEL"
ENV_ENACTMENT_BASE_URL = "REGULARS_ENACTMENT_BASE_URL"
ENV_ENACTMENT_API_KEY = "REGULARS_ENACTMENT_API_KEY"
ENV_SCORING_MODEL = "REGULARS_SCORING_MODEL"
ENV_CHEAP_MODEL = "REGULARS_CHEAP_MODEL"
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"


def llm_from_env(role: Role, env: Mapping[str, str] | None = None) -> LLMClient:
    """Construct the client for ``role``. ``env`` defaults to ``os.environ``."""
    e = os.environ if env is None else env
    api_key = e.get(ENV_ANTHROPIC_API_KEY) or None
    scoring_model = e.get(ENV_SCORING_MODEL) or DEFAULT_SCORING_MODEL

    if role == "scoring":
        return AnthropicLLM(scoring_model, api_key=api_key)
    if role == "cheap":
        return AnthropicLLM(e.get(ENV_CHEAP_MODEL) or DEFAULT_CHEAP_MODEL, api_key=api_key)

    base_url = e.get(ENV_ENACTMENT_BASE_URL) or None
    if base_url:
        return OpenAICompatibleLLM(
            e.get(ENV_ENACTMENT_MODEL) or CENTAUR_MODEL,
            base_url=base_url,
            api_key=e.get(ENV_ENACTMENT_API_KEY) or "EMPTY",
        )
    return AnthropicLLM(e.get(ENV_ENACTMENT_MODEL) or scoring_model, api_key=api_key)
