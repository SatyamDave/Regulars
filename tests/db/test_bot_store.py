"""One behavioural scenario, run against ``InMemoryBotStore`` and ``SqlBotStore`` alike.

``SqlBotStore`` (regulars/db/bot_store.py) is the SQL-backed ``regulars.bot.store.BotStore``:
it must behave exactly like the in-memory store the bot's own tests already exercise. Rather
than duplicating those tests, ``_run_scenario`` below exercises every ``BotStore`` method once
and is parametrized over both implementations.

The underlying ``groups``/``plans`` rows a real deployment would have (created by
``regulars.pipeline``) are stood up here directly with ``regulars.db.repo``, since
``SqlBotStore.save_group``/``save_plan`` write into rows keyed by a ``group_id``/``plan_id`` that
must already exist as foreign keys.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Engine

from regulars.bot.events import DirectedRating
from regulars.bot.onboarding import OnboardingState
from regulars.bot.plan import Slot, new_plan
from regulars.bot.store import BotStore, GroupChat, InMemoryBotStore, Member, MorningAfterSession
from regulars.db import repo
from regulars.db.bot_store import SqlBotStore
from regulars.db.engine import sqlite_memory_engine

T0 = datetime(2026, 9, 14, 17, 0, tzinfo=UTC)


def _run_scenario(store: BotStore, group_id: str, plan_id: str) -> None:
    # identity
    assert store.user_for_handle("+1555") is None
    u1 = store.ensure_user("+1555")
    assert store.ensure_user("+1555") == u1  # idempotent
    u2 = store.ensure_user("+1556")
    assert store.user_for_handle("+1555") == u1
    assert store.handle_for_user(u1) == "+1555"

    # groups
    assert store.get_group(group_id) is None
    assert store.group_for_chat("chat-1") is None
    members = (Member(u1, "+1555", "Priya"), Member(u2, "+1556", "Dev"))
    store.save_group(GroupChat(group_id=group_id, chat_id="chat-1", members=members))
    fetched = store.get_group(group_id)
    assert fetched is not None
    assert {m.user_id for m in fetched.members} == {u1, u2}
    assert store.group_for_chat("chat-1") is not None
    assert fetched.member(u1) is not None
    assert fetched.member_by_handle("+1556") is not None

    # removing a member persists
    store.save_group(GroupChat(group_id=group_id, chat_id="chat-1", members=(members[0],)))
    after_removal = store.get_group(group_id)
    assert after_removal is not None
    assert [m.user_id for m in after_removal.members] == [u1]

    # plans
    assert store.get_plan(plan_id) is None
    slot = Slot(id="s1", venue="Rooh", starts_at=T0 + timedelta(days=2))
    plan = new_plan(plan_id, group_id, [u1], [slot], T0 + timedelta(days=1))
    store.save_plan(plan)
    assert store.get_plan(plan_id) == plan
    assert plan in store.open_plans()
    assert store.open_plan_for_group(group_id) == plan
    assert store.open_plan_for_group("no-such-group") is None

    # onboarding
    assert store.get_onboarding(u1) is None
    state = OnboardingState(user_id=u1, stage="prompt_2")
    store.save_onboarding(state)
    assert store.get_onboarding(u1) == state

    # morning after
    assert store.get_morning_after(u1) is None
    session = MorningAfterSession(user_id=u1, plan_id=plan_id, group_id=group_id, others=(u2,))
    store.save_morning_after(session)
    assert store.get_morning_after(u1) == session
    store.clear_morning_after(u1)
    assert store.get_morning_after(u1) is None

    # ratings (persisted through the store's own save_ratings; no read-back method on the
    # Protocol, so this only asserts it does not raise)
    store.save_ratings(
        [DirectedRating(rater=u1, ratee=u2, would_see_again=True, felt_connected_1_7=6)], plan_id
    )

    # audit
    assert store.audit_for_user(u1) == []
    entry = store.write_audit(
        actor="bot",
        decision_type="group_opened",
        reason="two members",
        payload={"n": 2},
        subject_user_id=u1,
        group_id=group_id,
    )
    assert entry.reason == "two members"
    logged = store.audit_for_user(u1)
    assert len(logged) == 1
    assert logged[0].id == entry.id
    assert logged[0].payload == {"n": 2}


def test_in_memory_bot_store_scenario() -> None:
    _run_scenario(InMemoryBotStore(), group_id="g1", plan_id="p1")


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = sqlite_memory_engine()
    yield eng
    eng.dispose()


def test_sql_bot_store_scenario(engine: Engine) -> None:
    with engine.begin() as conn:
        group = repo.create_group(conn, zip_code="94110", assembly_method="model")
        plan = repo.create_plan(conn, group.id)
    _run_scenario(SqlBotStore(engine), group_id=group.id, plan_id=plan.id)
