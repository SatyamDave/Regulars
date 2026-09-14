from typing import Any

import pytest
from openai import Omit
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from regulars.core.llm import LLMClient, LLMMessage
from regulars.llm.openai_compat import CENTAUR_MODEL, OpenAICompatibleLLM


def completion(text: str) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=0,
        model=CENTAUR_MODEL,
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=text),
            )
        ],
    )


async def test_chat_completion_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = OpenAICompatibleLLM(CENTAUR_MODEL, base_url="http://vllm.local:8000/v1", api_key="x")
    assert isinstance(llm, LLMClient)
    calls: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs)
        return completion("  yeah.  ")

    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)
    out = await llm.complete(
        "SYS",
        [LLMMessage("user", "hi"), LLMMessage("assistant", "hey"), LLMMessage("user", "so")],
        temperature=0.95,
        max_tokens=160,
        seed=3,
    )
    assert out == "yeah."
    call = calls[0]
    assert call["model"] == CENTAUR_MODEL
    assert call["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hey"},
        {"role": "user", "content": "so"},
    ]
    assert call["temperature"] == 0.95
    assert call["max_tokens"] == 160
    assert call["seed"] == 3


async def test_seed_omitted_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = OpenAICompatibleLLM("m", base_url="http://x/v1")
    calls: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs)
        return completion("ok")

    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)
    await llm.complete("s", [LLMMessage("user", "hi")])
    assert isinstance(calls[0]["seed"], Omit)
    assert str(llm.client.base_url).rstrip("/") == "http://x/v1"
