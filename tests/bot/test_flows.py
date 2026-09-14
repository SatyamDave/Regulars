from __future__ import annotations

from datetime import timedelta

import pytest

from regulars.bot import voice
from regulars.bot.clock import FrozenClock
from regulars.bot.deposits import FakeDepositCollector
from regulars.bot.events import BackfillRequest, PruneRequest
from regulars.bot.flows import FlowError, MorningAfter, render_intro
from regulars.bot.plan import Plan, Slot, render_lock
from regulars.bot.runtime import BotRuntime
from regulars.bot.store import InMemoryBotStore, Member
from regulars.bot.transport import FakeTransport
from tests.bot.conftest import T0, dm, group_msg


async def open_group(
    runtime: BotRuntime, members: list[Member], hooks: dict[str, str], slots: list[Slot]
) -> Plan:
    return await runtime.flows.opener.open("g1", members, hooks, slots)


async def test_intro_then_immediately_the_proposal(
    runtime: BotRuntime,
    transport: FakeTransport,
    members: list[Member],
    hooks: dict[str, str],
    slots: list[Slot],
) -> None:
    plan = await open_group(runtime, members, hooks, slots)
    chat_id = transport.outbound[0].target
    assert transport.outbound[0].kind == "create_group"
    assert transport.groups[chat_id] == [m.handle for m in members]

    sent = transport.sent_to(chat_id)
    assert len(sent) == 2, "exactly one intro and one proposal, nothing in between"
    intro, proposal = sent
    for m in members:
        assert f"{m.name}: {hooks[m.user_id]}" in intro
    assert proposal.startswith("Three options.")
    assert "1. Rooh" in proposal and "2. Besharam" in proposal and "3. Ettan" in proposal
    assert plan.poll_open and plan.deadline == T0 + timedelta(hours=24)
    assert set(plan.commitment) == {m.user_id for m in members}

    for body in transport.all_bodies():
        assert "get to know each other" not in body.lower()
        assert "!" not in body


def test_intro_requires_a_hook_per_person(members: list[Member]) -> None:
    with pytest.raises(FlowError):
        render_intro(members, {members[0].user_id: "only one"})


async def test_votes_poll_deposits_drop_backfill_lock(
    runtime: BotRuntime,
    transport: FakeTransport,
    store: InMemoryBotStore,
    deposits: FakeDepositCollector,
    clock: FrozenClock,
    members: list[Member],
    hooks: dict[str, str],
    slots: list[Slot],
) -> None:
    plan = await open_group(runtime, members, hooks, slots)
    chat_id = transport.outbound[0].target

    # votes: 3 for Besharam (2), 2 for Rooh (1), one via venue name
    ballots = ["2", "2", "Besharam works for me", "1", "A"]
    for i, (m, ballot) in enumerate(zip(members, ballots, strict=False)):
        await runtime.process(group_msg(chat_id, m.handle, ballot, T0 + timedelta(minutes=i + 1)))
    current = store.get_plan(plan.id)
    assert current is not None
    assert current.votes == {"u1": "s2", "u2": "s2", "u3": "s2", "u4": "s1", "u5": "s1"}
    assert len(transport.sent_to(chat_id)) == 2, "the bot does not reply to votes"

    # nothing happens before the deadline
    await runtime.tick(clock.advance(timedelta(hours=12)))
    assert deposits.calls == []

    # deadline: poll closes, deposits requested, DM per member
    await runtime.tick(clock.advance(timedelta(hours=12)))
    current = store.get_plan(plan.id)
    assert current is not None
    assert current.locked_slot_id == "s2"
    assert len(deposits.calls) == 6
    assert all(
        c == (plan.id, u.user_id, 2000) for c, u in zip(deposits.calls, members, strict=True)
    )
    assert all(current.commitment[m.user_id] == "deposit_requested" for m in members)
    announce = transport.sent_to(chat_id)[-1]
    assert announce.startswith("Poll closed. Besharam, Fri Sep 18, 7pm.")
    for m in members:
        last_dm = transport.dms_to(m.handle)[-1]
        assert "$20 holds your seat" in last_dm and "https://pay.example/" in last_dm

    # five pay (webhook), one does not
    for m in members[:5]:
        await runtime.payment_received(plan.id, m.user_id)
    await runtime.tick(clock.advance(timedelta(hours=23)))
    current = store.get_plan(plan.id)
    assert current is not None and current.outcome == "open", "still inside the 24h window"
    assert runtime.events == []

    # 24h: unpaid dropped, backfill requested, then locked
    await runtime.tick(clock.advance(timedelta(hours=1)))
    final = store.get_plan(plan.id)
    assert final is not None
    assert final.outcome == "locked"
    assert final.commitment["u6"] == "dropped"
    assert final.members() == ["u1", "u2", "u3", "u4", "u5"]
    assert runtime.events == [
        BackfillRequest(group_id="g1", plan_id=plan.id, seats=1, dropped=("u6",))
    ]
    sam = members[5]
    assert sam.handle not in transport.groups[chat_id]
    assert transport.dms_to(sam.handle)[-1] == voice.DROPPED_DM.format(venue="Besharam")
    group = store.get_group("g1")
    assert group is not None and group.member("u6") is None

    lock_msg = transport.sent_to(chat_id)[-1]
    assert lock_msg == render_lock(final, {m.user_id: m.name for m in members})
    assert lock_msg.startswith("Locked. Besharam, Fri Sep 18, 7pm.")
    assert "Priya, Dev, Anika, Rohan and Meera are in." in lock_msg

    # removal is explainable (§10) from the audit row
    removal = [a for a in store.audit_for_user("u6") if a.decision_type == "member_removed"]
    assert len(removal) == 1 and removal[0].reason == voice.DROPPED_REASON
    await runtime.process(dm(sam.handle, "why was I removed?"))
    assert transport.dms_to(sam.handle)[-1].endswith(f"Straight answer: {voice.DROPPED_REASON}")


async def test_everyone_pays_locks_without_waiting_24h(
    runtime: BotRuntime,
    store: InMemoryBotStore,
    clock: FrozenClock,
    members: list[Member],
    hooks: dict[str, str],
    slots: list[Slot],
) -> None:
    plan = await open_group(runtime, members, hooks, slots)
    await runtime.tick(clock.advance(timedelta(hours=24)))
    for m in members:
        await runtime.payment_received(plan.id, m.user_id)
    await runtime.tick(clock.advance(timedelta(minutes=5)))
    final = store.get_plan(plan.id)
    assert (
        final is not None
        and final.outcome == "locked"
        and final.members() == [m.user_id for m in members]
    )
    assert runtime.events == []


async def test_credit_covers_deposit(
    transport: FakeTransport,
    store: InMemoryBotStore,
    clock: FrozenClock,
    members: list[Member],
    hooks: dict[str, str],
    slots: list[Slot],
) -> None:
    from regulars.bot.runtime import build_runtime

    deposits = FakeDepositCollector(credits={"u1": 2000, "u2": 500})
    runtime = build_runtime(transport, store, deposits, clock)
    plan = await open_group(runtime, members, hooks, slots)
    await runtime.tick(clock.advance(timedelta(hours=24)))
    current = store.get_plan(plan.id)
    assert current is not None
    assert current.commitment["u1"] == "paid"
    assert current.commitment["u2"] == "deposit_requested"
    assert transport.dms_to(members[0].handle)[-1].endswith("You are in.")
    assert (
        "Your credit covers $5 of the $20 deposit; $15 holds"
        in transport.dms_to(members[1].handle)[-1]
    )


# --- morning after -----------------------------------------------------------------------------


async def locked_plan(
    runtime: BotRuntime,
    store: InMemoryBotStore,
    clock: FrozenClock,
    members: list[Member],
    hooks: dict[str, str],
    slots: list[Slot],
) -> Plan:
    plan = await open_group(runtime, members, hooks, slots)
    await runtime.tick(clock.advance(timedelta(hours=24)))
    for m in members:
        await runtime.payment_received(plan.id, m.user_id)
    await runtime.tick(clock.now())
    final = store.get_plan(plan.id)
    assert final is not None and final.outcome == "locked"
    return final


async def test_morning_after_directed_rows_both_directions(
    runtime: BotRuntime,
    transport: FakeTransport,
    store: InMemoryBotStore,
    clock: FrozenClock,
    members: list[Member],
    hooks: dict[str, str],
    slots: list[Slot],
) -> None:
    plan = await locked_plan(runtime, store, clock, members, hooks, slots)
    chat_id = transport.outbound[0].target
    attended = ["u1", "u2", "u3", "u4", "u5"]
    clock.advance(timedelta(days=2))
    await runtime.flows.morning_after.start(plan, attended, forfeited=["u6"])

    receipt = transport.sent_to(chat_id)[-1]
    assert receipt.startswith("Last night. Rooh, Thu Sep 17, 7:30pm."), "no votes: earliest wins"
    assert "At the table: Priya, Dev, Anika, Rohan and Meera." in receipt
    assert "Sam did not make it. $4 each" in receipt

    priya, dev = members[0], members[1]
    prompt = transport.dms_to(priya.handle)[-1]
    assert prompt.startswith(voice.MORNING_AFTER_HEADER)
    assert "1. Dev" in prompt and "4. Meera" in prompt and "Priya" not in prompt.split("\n", 2)[2]
    assert transport.dms_to(members[5].handle)[-1] != prompt, "Sam did not attend; no prompt"

    # Priya: would see Dev and Anika again; Rohan was off; worth 6.
    await runtime.process(dm(priya.handle, "again: Dev, 2\noff: Rohan\nworth: 6"))
    # Dev: nobody again, no one off, worth 3 -- positional format, no labels.
    await runtime.process(dm(dev.handle, "none\nnone\n3"))

    rows = {(r.rater, r.ratee): r for _, r in store.ratings}
    assert len(rows) == 8, "4 rows per rater, two raters, never collapsed"
    assert rows[("u1", "u2")].would_see_again is True
    assert rows[("u1", "u3")].would_see_again is True
    assert rows[("u1", "u4")].would_see_again is False and rows[("u1", "u4")].was_off is True
    assert rows[("u1", "u5")].would_see_again is False
    assert all(rows[("u1", r)].felt_connected_1_7 == 6 for r in ("u2", "u3", "u4", "u5"))
    assert rows[("u2", "u1")].would_see_again is False, "Dev's view of Priya is its own row"
    assert rows[("u2", "u1")].felt_connected_1_7 == 3
    assert ("u1", "u1") not in rows and ("u2", "u2") not in rows

    prunes = [e for e in runtime.events if isinstance(e, PruneRequest)]
    assert [p.rater for p in prunes] == ["u1", "u2"]
    assert len(prunes[0].ratings) == 4 and prunes[0].plan_id == plan.id
    assert transport.dms_to(priya.handle)[-1] == voice.MORNING_AFTER_THANKS
    assert store.get_morning_after("u1") is None

    # an unreadable reply asks again and keeps the session open
    anika = members[2]
    await runtime.process(dm(anika.handle, "it was fine"))
    assert transport.dms_to(anika.handle)[-1] == voice.MORNING_AFTER_UNCLEAR
    assert store.get_morning_after("u3") is not None


@pytest.mark.parametrize(
    ("text", "again", "off", "worth"),
    [
        ("again: 1, 3\noff: none\nworth: 6", {"b", "d"}, set(), 6),
        ("Again - Dev and Rohan\nOff - Anika\nWorth - 7/7", {"b", "d"}, {"c"}, 7),
        ("1. everyone\n2. nobody\n3. 5", {"b", "c", "d"}, set(), 5),
        ("see again: dev\noff: no one", {"b"}, set(), None),
        ("again: 2 3", {"c", "d"}, set(), None),
    ],
)
def test_parse_reply_variants(text: str, again: set[str], off: set[str], worth: int | None) -> None:
    others = {"b": "Dev", "c": "Anika", "d": "Rohan"}
    rows = MorningAfter.parse_reply("a", text, others)
    assert {r.ratee for r in rows} == set(others)
    assert {r.ratee for r in rows if r.would_see_again} == again
    assert {r.ratee for r in rows if r.was_off} == off
    assert {r.felt_connected_1_7 for r in rows} == {worth}
    assert all(r.rater == "a" for r in rows)


def test_parse_reply_unreadable_is_empty() -> None:
    assert MorningAfter.parse_reply("a", "great night", {"b": "Dev"}) == []
