from __future__ import annotations

from datetime import UTC, datetime

import pytest

from regulars.bot.clock import FrozenClock
from regulars.bot.deposits import FakeDepositCollector
from regulars.bot.store import GroupChat, InMemoryBotStore, Member
from regulars.bot.tools import BotTools, ToolError
from regulars.bot.transport import FakeTransport
from regulars.core.llm import FakeLLM, LLMMessage
from regulars.llm.factory import _InstrumentedLLM, llm_from_env
from regulars.observability.instrumented import bot_tool_hook, llm_call_hook
from regulars.observability.metrics import BOT_TOOL_CALLS, BOT_TOOL_LATENCY, LLM_CALLS, LLM_LATENCY

T0 = datetime(2026, 9, 14, 17, 0, tzinfo=UTC)


@pytest.fixture
def store() -> InMemoryBotStore:
    s = InMemoryBotStore(now=lambda: T0)
    s.save_group(
        GroupChat(
            group_id="g1",
            chat_id="chat-1",
            members=(Member(user_id="u1", handle="+1", name="A"),),
        )
    )
    return s


async def test_bot_tool_hook_records_calls_and_latency(store: InMemoryBotStore) -> None:
    transport = FakeTransport()
    transport.groups["chat-1"] = ["+1"]
    tools = BotTools(
        transport, store, FakeDepositCollector(), FrozenClock(T0), on_call=bot_tool_hook()
    )

    before_calls = dict(BOT_TOOL_CALLS.samples())

    await tools.send_message("g1", "hi")

    after_calls = dict(BOT_TOOL_CALLS.samples())
    key = ("send_message", "ok")
    assert after_calls.get(key, 0) == before_calls.get(key, 0) + 1

    [(hist_key, _bucket_counts, _hist_sum, hist_count)] = [
        row for row in BOT_TOOL_LATENCY.samples() if row[0] == ("send_message",)
    ]
    assert hist_key == ("send_message",)
    assert hist_count >= 1


async def test_bot_tool_hook_records_error_status_and_reraises(store: InMemoryBotStore) -> None:
    transport = FakeTransport()
    tools = BotTools(
        transport, store, FakeDepositCollector(), FrozenClock(T0), on_call=bot_tool_hook()
    )

    before = dict(BOT_TOOL_CALLS.samples())

    with pytest.raises(ToolError):
        await tools.send_message("no-such-group", "hi")

    after = dict(BOT_TOOL_CALLS.samples())
    key = ("send_message", "error")
    assert after.get(key, 0) == before.get(key, 0) + 1


async def test_no_hook_means_no_metrics_dependency_and_no_behavior_change(
    store: InMemoryBotStore,
) -> None:
    transport = FakeTransport()
    transport.groups["chat-1"] = ["+1"]
    tools = BotTools(transport, store, FakeDepositCollector(), FrozenClock(T0))

    message_id = await tools.send_message("g1", "hi")

    assert message_id
    assert [a.decision_type for a in store.audit] == ["send_message"]


async def test_llm_call_hook_records_latency_by_role() -> None:
    hook = llm_call_hook("scoring")
    inner = FakeLLM(replies=["ok"])
    instrumented = _InstrumentedLLM(inner, hook)
    before = dict(LLM_CALLS.samples())

    result = await instrumented.complete("sys", [LLMMessage(role="user", content="hi")])

    assert result == "ok"
    after = dict(LLM_CALLS.samples())
    key = ("scoring", "ok")
    assert after.get(key, 0) == before.get(key, 0) + 1
    assert any(row[0] == ("scoring",) for row in LLM_LATENCY.samples())


async def test_llm_call_hook_records_error_status_and_reraises() -> None:
    hook = llm_call_hook("cheap")

    class _Boom:
        model = "boom-model"

        async def complete(self, *args: object, **kwargs: object) -> str:
            raise RuntimeError("llm exploded")

    instrumented = _InstrumentedLLM(_Boom(), hook)  # type: ignore[arg-type]
    before = dict(LLM_CALLS.samples())

    with pytest.raises(RuntimeError):
        await instrumented.complete("sys", [])

    after = dict(LLM_CALLS.samples())
    key = ("cheap", "error")
    assert after.get(key, 0) == before.get(key, 0) + 1


def test_llm_from_env_on_call_wraps_and_reports() -> None:
    hook = llm_call_hook("cheap")
    client = llm_from_env("cheap", {"ANTHROPIC_API_KEY": "test"}, on_call=hook)
    assert isinstance(client, _InstrumentedLLM)
    assert client.model  # the wrapper exposes the inner model id


def test_llm_from_env_without_on_call_returns_plain_client() -> None:
    client = llm_from_env("cheap", {"ANTHROPIC_API_KEY": "test"})
    assert not isinstance(client, _InstrumentedLLM)
