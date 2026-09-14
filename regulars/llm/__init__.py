"""LLMClient providers: Anthropic (scoring/cheap) and OpenAI-compatible (Centaur enactment)."""

from regulars.llm.anthropic_client import AnthropicLLM, UsageTotals, supports_temperature
from regulars.llm.factory import DEFAULT_CHEAP_MODEL, DEFAULT_SCORING_MODEL, Role, llm_from_env
from regulars.llm.openai_compat import CENTAUR_MODEL, OpenAICompatibleLLM

__all__ = [
    "CENTAUR_MODEL",
    "DEFAULT_CHEAP_MODEL",
    "DEFAULT_SCORING_MODEL",
    "AnthropicLLM",
    "OpenAICompatibleLLM",
    "Role",
    "UsageTotals",
    "llm_from_env",
    "supports_temperature",
]
