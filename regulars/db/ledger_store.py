"""``SqlLedgerStore``: ``regulars.payments.ledger.LedgerStore`` over ``deposit_credits`` +
``deposit_holds`` (0002).

``LedgerStore`` methods are ``async`` (the ledger is used from the async bot/payments path);
each opens its own synchronous transaction with ``engine.begin()``, mirroring ``SqlBotStore``.
"""

from __future__ import annotations

from sqlalchemy import Engine

from regulars.db import repo
from regulars.payments.ledger import HoldRecord, LedgerEntry


class SqlLedgerStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    async def add_entry(self, entry: LedgerEntry) -> None:
        with self.engine.begin() as conn:
            repo.insert_deposit_credit(
                conn,
                user_id=entry.user_id,
                amount_cents=entry.amount_cents,
                kind=entry.kind,
                plan_id=entry.plan_id,
            )

    async def entries(
        self, *, user_id: str | None = None, plan_id: str | None = None
    ) -> list[LedgerEntry]:
        with self.engine.connect() as conn:
            rows = repo.deposit_credits_query(conn, user_id=user_id, plan_id=plan_id)
        return [
            LedgerEntry(
                id=r.id,
                user_id=r.user_id,
                kind=r.kind,
                amount_cents=r.amount_cents,
                plan_id=r.plan_id,
                created_at=r.created_at,
            )
            for r in rows
        ]

    async def put_hold(self, hold: HoldRecord) -> None:
        with self.engine.begin() as conn:
            repo.put_deposit_hold(
                conn,
                plan_id=hold.plan_id,
                user_id=hold.user_id,
                intent_id=hold.intent_id,
                amount_cents=hold.amount_cents,
                credit_cents=hold.credit_cents,
                state=hold.state,
            )

    async def holds(self, plan_id: str) -> list[HoldRecord]:
        with self.engine.connect() as conn:
            rows = repo.deposit_holds_for_plan(conn, plan_id)
        return [
            HoldRecord(
                plan_id=r.plan_id,
                user_id=r.user_id,
                intent_id=r.intent_id,
                amount_cents=r.amount_cents,
                credit_cents=r.credit_cents,
                state=r.state,
            )
            for r in rows
        ]
