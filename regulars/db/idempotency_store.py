"""``SqlIdempotencyStore``: ``regulars.payments.idempotency.WebhookIdempotencyStore`` over
``processed_webhook_events`` (0003).

Same shape as ``regulars.db.ledger_store.SqlLedgerStore``: the Protocol and in-memory
implementation live in the owning package (``regulars.payments.idempotency``); the SQL adapter
lives here so ``regulars.payments`` does not import ``regulars.db`` (``regulars.core`` is the
only cross-package import surface — ``regulars.db`` implementing another package's storage
protocol is the established exception, not the payments package reaching into ``regulars.db``).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Engine

from regulars.db import repo


class SqlIdempotencyStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    async def already_processed(self, event_id: str) -> bool:
        with self.engine.connect() as conn:
            row = repo.get_processed_webhook_event(conn, event_id)
        return row is not None and row.status == "done"

    async def mark_processing(self, event_id: str, event_type: str, now: datetime) -> None:
        with self.engine.begin() as conn:
            existing = repo.get_processed_webhook_event(conn, event_id)
            if existing is not None and existing.status == "done":
                return
            repo.put_processed_webhook_event(
                conn,
                event_id=event_id,
                event_type=event_type,
                received_at=existing.received_at if existing is not None else now,
                status="processing",
            )

    async def mark_done(self, event_id: str, now: datetime) -> None:
        with self.engine.begin() as conn:
            existing = repo.get_processed_webhook_event(conn, event_id)
            received_at = existing.received_at if existing is not None else now
            event_type = existing.event_type if existing is not None else ""
            repo.put_processed_webhook_event(
                conn,
                event_id=event_id,
                event_type=event_type,
                received_at=received_at,
                status="done",
                processed_at=now,
                error=None,
            )

    async def mark_failed(self, event_id: str, now: datetime, error: str) -> None:
        with self.engine.begin() as conn:
            existing = repo.get_processed_webhook_event(conn, event_id)
            received_at = existing.received_at if existing is not None else now
            event_type = existing.event_type if existing is not None else ""
            repo.put_processed_webhook_event(
                conn,
                event_id=event_id,
                event_type=event_type,
                received_at=received_at,
                status="failed",
                processed_at=now,
                error=error,
            )
