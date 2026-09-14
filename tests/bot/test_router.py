from __future__ import annotations

from datetime import timedelta

from regulars.bot import voice
from regulars.bot.events import LegibilityRequest
from regulars.bot.onboarding import OnboardingState
from regulars.bot.plan import Slot, new_plan
from regulars.bot.router import Emit, LogSignal, SendMessage, parse_vote
from regulars.bot.runtime import BotRuntime
from regulars.bot.store import GroupChat, InMemoryBotStore, Member
from regulars.bot.transport import FakeTransport
from tests.bot.conftest import T0, dm, group_msg

HANDLE = "+14155550100"


def onboarded(store: InMemoryBotStore, handle: str = HANDLE) -> str:
    user = store.ensure_user(handle)
    store.save_onboarding(OnboardingState(user_id=user, stage="done"))
    return user


async def test_legibility_command_emits_request(
    runtime: BotRuntime, store: InMemoryBotStore, transport: FakeTransport
) -> None:
    user = onboarded(store)
    actions = await runtime.process(dm(HANDLE, "hey, what do you know about me?"))
    assert actions == [
        SendMessage(voice.LEGIBILITY_ACK, to_user=user),
        Emit(LegibilityRequest(user_id=user)),
    ]
    assert runtime.events == [LegibilityRequest(user_id=user)]
    assert transport.dms_to(HANDLE) == [voice.LEGIBILITY_ACK]


async def test_removal_question_without_removal(
    runtime: BotRuntime, store: InMemoryBotStore, transport: FakeTransport
) -> None:
    onboarded(store)
    await runtime.process(dm(HANDLE, "why was i removed"))
    assert transport.dms_to(HANDLE) == [voice.NOT_REMOVED]


async def test_removal_question_answers_with_latest_audit_reason(
    runtime: BotRuntime, store: InMemoryBotStore, transport: FakeTransport
) -> None:
    user = onboarded(store)
    store.write_audit(
        actor="bot", decision_type="member_removed", reason="older reason", subject_user_id=user
    )
    store.now = lambda: T0 + timedelta(days=1)
    store.write_audit(
        actor="bot",
        decision_type="member_removed",
        reason="Two of five said they would not see you again.",
        subject_user_id=user,
    )
    await runtime.process(dm(HANDLE, "Why was I removed from the group?"))
    reply = transport.dms_to(HANDLE)[-1]
    assert reply == voice.REMOVAL_REASON.format(
        when="Tue Sep 15, 5pm", reason="Two of five said they would not see you again."
    )


async def test_group_messages_log_latency_and_tapbacks_and_votes(
    runtime: BotRuntime, store: InMemoryBotStore, members: list[Member], slots: list[Slot]
) -> None:
    group = GroupChat(group_id="g1", chat_id="chat-1", members=tuple(members))
    store.save_group(group)
    plan = new_plan("p1", "g1", [m.user_id for m in members], slots, T0 + timedelta(days=1))
    store.save_plan(plan)
    a, b = members[0], members[1]

    first = await runtime.process(group_msg("chat-1", a.handle, "2", T0))
    assert first == [], "first message has nothing to measure latency against"

    second = await runtime.process(group_msg("chat-1", b.handle, "1", T0 + timedelta(seconds=90)))
    assert second == [LogSignal("u2", "reply_latency_ms", 90_000.0, "g1")]

    again = await runtime.process(
        group_msg("chat-1", b.handle, "actually 3", T0 + timedelta(seconds=100))
    )
    assert again == [], "a follow-up by the same sender is not a reply"

    tap = await runtime.process(
        group_msg("chat-1", a.handle, "", T0 + timedelta(seconds=120), tapback="❤️")
    )
    assert tap == [LogSignal("u1", "tapback", 1.0, group_id="g1")]

    stranger = await runtime.process(group_msg("chat-1", "+19999999999", "1", T0))
    assert stranger == []

    current = store.get_plan("p1")
    assert current is not None
    assert current.votes == {"u1": "s2", "u2": "s1"}, "'actually 3' is chatter, not a ballot"
    assert [x.decision_type for x in store.audit] == ["log_signal", "log_signal"]


def test_parse_vote(slots: list[Slot]) -> None:
    plan = new_plan("p", "g", ["a"], slots, T0)
    assert parse_vote("1", plan) == "s1"
    assert parse_vote(" 2. ", plan) == "s2"
    assert parse_vote("C", plan) == "s3"
    assert parse_vote("option 2", plan) == "s2"
    assert parse_vote("3 works", plan) == "s3"
    assert parse_vote("Ettan please", plan) == "s3"
    assert parse_vote("rooh or besharam, either", plan) is None
    assert parse_vote("9", plan) is None
    assert parse_vote("see you all there", plan) is None
