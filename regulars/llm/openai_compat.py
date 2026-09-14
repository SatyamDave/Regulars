"""OpenAI-compatible ``LLMClient`` for self-served open-weight models (§2 base model).

Intended target: **Centaur** (``marcelbinz/Llama-3.1-Centaur-70B``) served by vLLM or TGI behind
their OpenAI-compatible chat-completions endpoint. Centaur is used for persona *enactment* only,
never for scoring: it is finetuned on human decision data (Psych-101) and gives human-shaped
behavior under pressure, not conversational competence. Scoring goes through the hosted frontier
model (``regulars.llm.anthropic_client``).

``seed`` is passed straight through; vLLM honors it for reproducible sampling.
"""

from __future__ import annotations

from collections.abc import Sequence

from openai import AsyncOpenAI, Omit, omit
from openai.types.chat import ChatCompletionMessageParam

from regulars.core.llm import LLMMessage

CENTAUR_MODEL = "marcelbinz/Llama-3.1-Centaur-70B"


class OpenAICompatibleLLM:
    """``LLMClient`` over ``/v1/chat/completions`` on any OpenAI-compatible server."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        *,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        # vLLM/TGI ignore the key unless configured; the SDK still requires a non-empty string.
        self.client = client or AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def complete(
        self,
        system: str,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> str:
        chat: list[ChatCompletionMessageParam] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "user":
                chat.append({"role": "user", "content": m.content})
            else:
                chat.append({"role": "assistant", "content": m.content})
        seed_param: int | Omit = seed if seed is not None else omit
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=chat,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed_param,
        )
        if not response.choices:
            return ""
        return (response.choices[0].message.content or "").strip()
