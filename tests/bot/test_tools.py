from __future__ import annotations

import inspect
from datetime import timedelta

import pytest

from regulars.bot.deposits import FakeDepositCollector
from regulars.bot.plan import Slot, close_poll
from regulars.bot.runtime import BotRuntime
from regulars.bot.store import GroupChat, InMemoryBotStore, Member
from regulars.bot.tools import TOOL_NAMES, BotTools, ToolError
from regulars.bot.transport import FakeTransport
from tests.bot.conftest import T0


def test_exactly_five_tools() -> None:
    assert len(BotTools.tools()) == 5
    assert BotTools.tools() == TOOL_NAMES
    public_async = [
        name
        for name, member in inspect.getmembers(BotTools, inspect.iscoroutinefunction)
        if not name.startswith("_")
    ]
    assert sorted(public_async) == sorted(TOOL_NAMES)


@pytest.fixture
def group(store: InMemoryBotStore, members: list[Member], transport: FakeTransport) -> GroupChat:
    g = GroupChat(group_id="g1", chat_id="chat-1", members=tuple(members))
    store.save_group(g)
    transport.groups["chat-1"] = [m.handle for m in members]
    return g


async def test_every_tool_call_writes_one_audit_row(
    runtime: BotRuntime,
    store: InMemoryBotStore,
    transport: FakeTransport,
    deposits: FakeDepositCollector,
    group: GroupChat,
    slots: list[Slot],
) -> None:
    tools = runtime.tools
    assert store.audit == []

    await tools.send_message("g1", "hello table")
    assert [a.decision_type for a in store.audit] == ["send_message"]
    assert transport.sent_to("chat-1") == ["hello table"]

    await tools.send_message("g1", "just you", to_user="u1")
    assert store.audit[-1].subject_user_id == "u1"
    assert transport.dms_to(group.members[0].handle) == ["just you"]

    plan = await tools.create_plan("g1", slots, T0 + timedelta(days=1))
    assert store.audit[-1].decision_type == "create_plan"
    assert store.audit[-1].plan_id == plan.id
    assert store.get_plan(plan.id) == plan

    with pytest.raises(ToolError):
        await tools.collect_deposit(plan.id, "u1", 2000)
    assert deposits.calls == [], "no hold is opened before the poll has closed"

    store.save_plan(close_poll(plan, T0))
    req = await tools.collect_deposit(plan.id, "u1", 2000)
    assert deposits.calls == [(plan.id, "u1", 2000)]
    assert store.audit[-1].decision_type == "collect_deposit"
    assert store.audit[-1].payload["intent_id"] == req.intent_id
    stored = store.get_plan(plan.id)
    assert stored is not None and stored.commitment["u1"] == "deposit_requested"

    await tools.remove_member("g1", "u6", "did not pay")
    assert store.audit[-1].decision_type == "member_removed"
    assert store.audit[-1].reason == "did not pay"
    assert "+15550000006" not in transport.groups["chat-1"]
    g = store.get_group("g1")
    assert g is not None and g.member("u6") is None

    await tools.log_signal("u2", "reply_latency_ms", 4200.0, group_id="g1")
    assert store.audit[-1].decision_type == "log_signal"
    assert store.audit[-1].payload == {"signal_type": "reply_latency_ms", "value": 4200.0}

    assert len(store.audit) == 6  # five distinct tools, send_message twice
    assert all(a.actor == "bot" and a.created_at == T0 for a in store.audit)


async def test_tools_refuse_unknown_targets(runtime: BotRuntime) -> None:
    with pytest.raises(ToolError):
        await runtime.tools.send_message("nope", "x")
    with pytest.raises(ToolError):
        await runtime.tools.send_message(None, "x", to_user="ghost")
    with pytest.raises(ToolError):
        await runtime.tools.collect_deposit("nope", "u1", 2000)
