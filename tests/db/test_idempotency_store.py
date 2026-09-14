"""One behavioural scenario, run against ``InMemoryIdempotencyStore`` and ``SqlIdempotencyStore``
alike, mirroring the ``InMemoryBotStore``/``SqlBotStore`` pattern in ``tests/db/test_bot_store.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Engine

from regulars.db.engine import sqlite_memory_engine
from regulars.db.idempotency_store import SqlIdempotencyStore
from regulars.payments.idempotency import InMemoryIdempotencyStore, WebhookIdempotencyStore

T0 = datetime(2026, 9, 14, 17, 0, tzinfo=UTC)


async def _run_scenario(store: WebhookIdempotencyStore) -> None:
    assert await store.already_processed("evt_1") is False

    await store.mark_processing("evt_1", "payment_intent.amount_capturable_updated", T0)
    assert await store.already_processed("evt_1") is False  # processing, not done yet

    await store.mark_done("evt_1", T0 + timedelta(seconds=1))
    assert await store.already_processed("evt_1") is True

    # A redelivery after completion does not regress the status back to "processing".
    await store.mark_processing("evt_1", "payment_intent.amount_capturable_updated", T0)
    assert await store.already_processed("evt_1") is True

    # A second, distinct event fails and is recorded, independent of the first.
    await store.mark_processing("evt_2", "payment_intent.canceled", T0)
    await store.mark_failed("evt_2", T0 + timedelta(seconds=1), "boom")
    assert await store.already_processed("evt_2") is False
    assert await store.already_processed("evt_1") is True  # unaffected


async def test_in_memory_idempotency_store_scenario() -> None:
    await _run_scenario(InMemoryIdempotencyStore())


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = sqlite_memory_engine()
    yield eng
    eng.dispose()


async def test_sql_idempotency_store_scenario(engine: Engine) -> None:
    await _run_scenario(SqlIdempotencyStore(engine))
