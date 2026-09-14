"""``SqlLedgerStore`` against a real SQLite engine: same contract as ``InMemoryLedgerStore``
(exercised in tests/payments/test_ledger.py), backed by ``deposit_credits`` + ``deposit_holds``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine

from regulars.db import repo
from regulars.db.engine import sqlite_memory_engine
from regulars.db.ledger_store import SqlLedgerStore
from regulars.payments.ledger import Ledger


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = sqlite_memory_engine()
    yield eng
    eng.dispose()


@pytest.fixture
def plan_and_users(engine: Engine) -> tuple[str, str, str]:
    with engine.begin() as conn:
        u1 = repo.create_user(conn, phone_hash="h1", zip_code="94110")
        u2 = repo.create_user(conn, phone_hash="h2", zip_code="94110")
        group = repo.create_group(conn, zip_code="94110", assembly_method="model")
        plan = repo.create_plan(conn, group.id)
    return plan.id, u1.id, u2.id


async def test_hold_settle_and_redistribute(
    engine: Engine, plan_and_users: tuple[str, str, str]
) -> None:
    plan_id, u1, u2 = plan_and_users
    store = SqlLedgerStore(engine)
    ledger = Ledger(store)

    await ledger.hold(plan_id, u1, "pi_1", 2000)
    await ledger.hold(plan_id, u2, "pi_2", 2000)

    settlement = await ledger.settle_plan(plan_id, attended=[u1], paid=[u1, u2])
    assert settlement.forfeited_cents == 2000
    assert settlement.credits[u1] == 1800  # 2000 - 10% platform cut
    assert await ledger.available_credit(u1) == 1800

    applied = await ledger.apply_credit(u1, plan_id, 500)
    assert applied == 500
    assert await ledger.available_credit(u1) == 1300

    holds = await store.holds(plan_id)
    assert {h.user_id: h.state for h in holds} == {u1: "released", u2: "captured"}


async def test_entries_filter_by_plan_and_user(
    engine: Engine, plan_and_users: tuple[str, str, str]
) -> None:
    plan_id, u1, u2 = plan_and_users
    store = SqlLedgerStore(engine)
    ledger = Ledger(store)
    await ledger.hold(plan_id, u1, "pi_1", 2000)
    await ledger.hold(plan_id, u2, "pi_2", 1500)

    by_user = await store.entries(user_id=u1)
    assert {e.plan_id for e in by_user} == {plan_id}
    assert sum(e.amount_cents for e in by_user) == 2000

    by_plan = await store.entries(plan_id=plan_id)
    assert {e.user_id for e in by_plan} == {u1, u2}
