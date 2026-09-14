from typing import Any

import httpx2
import pytest
from anthropic import BadRequestError, InternalServerError, RateLimitError
from anthropic.types import Message, TextBlock, Usage

from regulars.core.llm import LLMClient, LLMMessage
from regulars.llm.anthropic_client import AnthropicLLM, supports_temperature


def fake_message(text: str = "hello", *, cache_read: int = 0) -> Message:
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-haiku-4-5",
        content=[TextBlock(type="text", text=text)],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(
            input_tokens=10,
            output_tokens=3,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=0,
        ),
    )


def status_error(cls: type, status: int, headers: dict[str, str] | None = None) -> Exception:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request, headers=headers or {})
    return cls("boom", response=response, body=None)


class Recorder:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, Message)
        return outcome


def patched(
    monkeypatch: pytest.MonkeyPatch, outcomes: list[Any], **kwargs: Any
) -> tuple[AnthropicLLM, Recorder]:
    llm = AnthropicLLM(kwargs.pop("model", "claude-haiku-4-5"), api_key="test-key", **kwargs)
    recorder = Recorder(outcomes)
    monkeypatch.setattr(llm.client.messages, "create", recorder)
    return llm, recorder


def test_is_llm_client() -> None:
    assert isinstance(AnthropicLLM("claude-haiku-4-5", api_key="k"), LLMClient)


async def test_system_sent_as_cached_block(monkeypatch: pytest.MonkeyPatch) -> None:
    llm, rec = patched(monkeypatch, [fake_message()])
    out = await llm.complete("PERSONA PREFIX", [LLMMessage("user", "hi")], temperature=0.95)
    assert out == "hello"
    call = rec.calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert call["system"] == [
        {"type": "text", "text": "PERSONA PREFIX", "cache_control": {"type": "ephemeral"}}
    ]
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    assert call["extra_body"] == {"temperature": 0.95}
    assert call["max_tokens"] == 512


async def test_cache_system_false_sends_plain_string(monkeypatch: pytest.MonkeyPatch) -> None:
    llm, rec = patched(monkeypatch, [fake_message()], cache_system=False)
    await llm.complete("sys", [LLMMessage("user", "hi")])
    assert rec.calls[0]["system"] == "sys"


def test_supports_temperature_by_model_family() -> None:
    assert supports_temperature("claude-haiku-4-5")
    assert supports_temperature("claude-opus-4-6")
    assert not supports_temperature("claude-opus-5")
    assert not supports_temperature("claude-sonnet-5")
    assert not supports_temperature("claude-fable-5-1")


async def test_temperature_omitted_on_models_that_reject_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm, rec = patched(monkeypatch, [fake_message()], model="claude-opus-5")
    await llm.complete("sys", [LLMMessage("user", "hi")], temperature=0.95)
    assert rec.calls[0]["extra_body"] is None
    assert "temperature" not in rec.calls[0]


async def test_usage_totals_accumulate(monkeypatch: pytest.MonkeyPatch) -> None:
    llm, _ = patched(monkeypatch, [fake_message(cache_read=7), fake_message(cache_read=9)])
    await llm.complete("sys", [LLMMessage("user", "a")])
    await llm.complete("sys", [LLMMessage("user", "b")])
    assert llm.usage_totals.calls == 2
    assert llm.usage_totals.input_tokens == 20
    assert llm.usage_totals.output_tokens == 6
    assert llm.usage_totals.cache_read_tokens == 16


async def test_retries_rate_limit_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    llm, rec = patched(
        monkeypatch,
        [
            status_error(RateLimitError, 429),
            status_error(RateLimitError, 429, {"retry-after": "0.25"}),
            status_error(InternalServerError, 529),
            fake_message("finally"),
        ],
    )
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    llm.sleep = fake_sleep
    assert await llm.complete("sys", [LLMMessage("user", "hi")]) == "finally"
    assert len(rec.calls) == 4
    assert len(slept) == 3
    assert slept[1] == 0.25  # retry-after honored
    assert 0.5 <= slept[0] <= 1.5  # jittered around base delay
    assert 1.0 <= slept[2] <= 6.0  # attempt 2: 4s ceiling, jittered


async def test_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    llm, rec = patched(monkeypatch, [status_error(RateLimitError, 429)] * 6, max_retries=5)

    async def no_sleep(_: float) -> None:
        return None

    llm.sleep = no_sleep
    with pytest.raises(RateLimitError):
        await llm.complete("sys", [LLMMessage("user", "hi")])
    assert len(rec.calls) == 6  # 1 + 5 retries


async def test_client_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    llm, rec = patched(monkeypatch, [status_error(BadRequestError, 400)])
    with pytest.raises(BadRequestError):
        await llm.complete("sys", [LLMMessage("user", "hi")])
    assert len(rec.calls) == 1
