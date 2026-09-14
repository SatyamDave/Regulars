from __future__ import annotations

from datetime import timedelta

import pytest

from regulars.bot.plan import (
    Plan,
    PlanError,
    Slot,
    close_poll,
    drop_unpaid,
    format_money,
    format_when,
    lock,
    mark_deposit_requested,
    mark_paid,
    new_plan,
    render_lock,
    render_proposal,
    render_receipt,
    tied,
    vote,
)
from tests.bot.conftest import T0

USERS = ["a", "b", "c", "d"]
NAMES = {"a": "Priya", "b": "Dev", "c": "Anika", "d": "Rohan"}


def make(slots: list[Slot]) -> Plan:
    return new_plan("p1", "g1", USERS, slots, deadline=T0 + timedelta(days=1))


def test_vote_records_and_overrides(slots: list[Slot]) -> None:
    p = vote(make(slots), "a", "s1")
    p = vote(p, "a", "s2")
    assert p.votes == {"a": "s2"}


def test_vote_rejects_unknown_slot_and_non_member(slots: list[Slot]) -> None:
    with pytest.raises(PlanError):
        vote(make(slots), "a", "nope")
    with pytest.raises(PlanError):
        vote(make(slots), "zed", "s1")


def test_close_poll_plurality(slots: list[Slot]) -> None:
    p = make(slots)
    for u, s in (("a", "s2"), ("b", "s2"), ("c", "s3")):
        p = vote(p, u, s)
    closed = close_poll(p, T0)
    assert closed.locked_slot_id == "s2"
    assert closed.poll_closed_at == T0
    assert not closed.poll_open
    assert not tied(closed)
    with pytest.raises(PlanError):
        vote(closed, "d", "s1")


def test_close_poll_tie_goes_to_earliest(slots: list[Slot]) -> None:
    p = make(slots)
    p = vote(vote(p, "a", "s3"), "b", "s2")  # s2 starts before s3
    closed = close_poll(p, T0)
    assert closed.locked_slot_id == "s2"
    assert tied(closed)


def test_close_poll_no_votes_picks_earliest(slots: list[Slot]) -> None:
    assert close_poll(make(slots), T0).locked_slot_id == "s1"


def test_deposit_then_drop_unpaid_after_24h(slots: list[Slot]) -> None:
    p = close_poll(make(slots), T0)
    for u in USERS:
        p = mark_deposit_requested(p, u, f"pi_{u}", T0)
    p = mark_paid(mark_paid(p, "a"), "b")
    same, dropped = drop_unpaid(p, T0 + timedelta(hours=23))
    assert dropped == [] and same == p
    later, dropped = drop_unpaid(p, T0 + timedelta(hours=24))
    assert dropped == ["c", "d"]
    assert later.members() == ["a", "b"]
    assert later.all_paid()
    locked = lock(later, T0 + timedelta(hours=24))
    assert locked.outcome == "locked"


def test_lock_requires_closed_poll_and_full_payment(slots: list[Slot]) -> None:
    with pytest.raises(PlanError):
        lock(make(slots), T0)
    closed = close_poll(make(slots), T0)
    with pytest.raises(PlanError):
        lock(closed, T0)


def test_mark_paid_is_idempotent(slots: list[Slot]) -> None:
    p = mark_paid(close_poll(make(slots), T0), "a")
    assert mark_paid(p, "a") == p


def test_rendering_is_a_pure_function_of_the_plan(slots: list[Slot]) -> None:
    p = make(slots)
    a, b = render_proposal(p, NAMES), render_proposal(p.model_copy(), NAMES)
    assert a == b
    assert "1. Rooh, Thu Sep 17, 7:30pm" in a
    assert "2. Besharam, Fri Sep 18, 7pm" in a
    assert "$20" in a
    assert "!" not in a


def test_render_lock_lists_paid_members(slots: list[Slot]) -> None:
    p = close_poll(make(slots), T0)
    for u in USERS:
        p = mark_paid(mark_deposit_requested(p, u, f"pi_{u}", T0), u)
    p = lock(p, T0)
    text = render_lock(p, NAMES)
    assert text.startswith("Locked. Rooh, Thu Sep 17, 7:30pm.")
    assert "Priya, Dev, Anika and Rohan are in." in text


def test_render_receipt_splits_forfeit(slots: list[Slot]) -> None:
    p = close_poll(make(slots), T0)
    text = render_receipt(p, NAMES, attended=["a", "b"], forfeited=["c"])
    assert "At the table: Priya and Dev." in text
    assert "Anika did not make it. $10 each" in text
    all_showed = render_receipt(p, NAMES, attended=USERS, forfeited=[])
    assert "4 invited, 4 showed." in all_showed


def test_formatters() -> None:
    assert format_money(2000) == "$20"
    assert format_money(1250) == "$12.50"
    assert format_when(T0) == "Mon Sep 14, 5pm"
