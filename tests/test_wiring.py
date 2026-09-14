"""``regulars.wiring.build_system``: a redelivered Stripe webhook must not double-process a
payment (Stripe's own docs guarantee at-least-once delivery, retrying on any non-2xx or timeout).
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from regulars.app import Settings
from regulars.bot.clock import FrozenClock
from regulars.bot.transport import FakeTransport
from regulars.db.engine import sqlite_memory_engine
from regulars.wiring import build_system, fake_stripe_event

T0 = datetime(2026, 9, 14, 17, 0, tzinfo=UTC)


async def test_replayed_stripe_webhook_triggers_payment_received_once() -> None:
    clock = FrozenClock(T0)
    engine = sqlite_memory_engine()
    system = build_system(Settings(), engine=engine, transport=FakeTransport(), clock=clock)
    client = TestClient(system.app)

    # Replace payment_received outright (rather than wrapping it): this test is about the
    # webhook-dispatch idempotency wrapper, not the full plan/rsvp setup ``record_payment``
    # otherwise needs (that path is covered end-to-end in ``tests/test_e2e.py``).
    calls: list[tuple[str, str]] = []

    async def counting_payment_received(plan_id: str, user_id: str):  # type: ignore[no-untyped-def]
        calls.append((plan_id, user_id))

    system.runtime.payment_received = counting_payment_received  # type: ignore[method-assign]

    body = fake_stripe_event(
        "payment_intent.amount_capturable_updated",
        plan_id="plan-1",
        user_id="user-1",
        intent_id="pi_1",
    )
    headers = {"Stripe-Signature": "test-does-not-check"}

    # Stripe redelivers the identical event id (same body) up to three times here.
    for _ in range(3):
        r = client.post("/webhooks/stripe", content=body, headers=headers)
        assert r.status_code == 200

    assert calls == [("plan-1", "user-1")]
