"""``IdempotentPaymentEventSink`` + ``InMemoryIdempotencyStore``: a replayed Stripe event id is a
no-op, and a failing handler is recorded as ``failed`` rather than silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from regulars.payments.idempotency import IdempotentPaymentEventSink, InMemoryIdempotencyStore
from regulars.payments.stripe_deposits import StripeEvent

T0 = datetime(2026, 9, 14, 17, 0, tzinfo=UTC)


@dataclass
class RecordingSink:
    """Stands in for ``regulars.wiring``'s ``_PaymentSink`` -> ``runtime.payment_received``."""

    events: list[StripeEvent] = field(default_factory=list)
    fail_on: set[str] = field(default_factory=set)

    async def handle(self, event: StripeEvent) -> None:
        if event.id in self.fail_on:
            raise RuntimeError(f"boom processing {event.id}")
        self.events.append(event)


def _event(
    event_id: str = "evt_1", kind: str = "payment_intent.amount_capturable_updated"
) -> StripeEvent:
    return StripeEvent(id=event_id, type=kind, object={"id": "pi_1"})


async def test_replayed_event_id_is_a_no_op() -> None:
    inner = RecordingSink()
    sink = IdempotentPaymentEventSink(inner, InMemoryIdempotencyStore(), clock=lambda: T0)

    event = _event()
    await sink.handle(event)
    await sink.handle(event)  # Stripe redelivers the same event id
    await sink.handle(event)

    assert inner.events == [event]  # handled exactly once


async def test_two_different_event_ids_both_processed() -> None:
    inner = RecordingSink()
    sink = IdempotentPaymentEventSink(inner, InMemoryIdempotencyStore(), clock=lambda: T0)

    await sink.handle(_event("evt_1"))
    await sink.handle(_event("evt_2"))

    assert [e.id for e in inner.events] == ["evt_1", "evt_2"]


async def test_failing_handler_is_recorded_as_failed_and_reraised() -> None:
    inner = RecordingSink(fail_on={"evt_1"})
    store = InMemoryIdempotencyStore()
    sink = IdempotentPaymentEventSink(inner, store, clock=lambda: T0)

    with pytest.raises(RuntimeError, match="boom processing evt_1"):
        await sink.handle(_event("evt_1"))

    record = store.get("evt_1")
    assert record is not None
    assert record.status == "failed"
    assert record.error == "boom processing evt_1"
    assert store.failed_events() == [record]
    assert inner.events == []


async def test_failed_event_can_be_retried_and_then_succeeds() -> None:
    inner = RecordingSink(fail_on={"evt_1"})
    store = InMemoryIdempotencyStore()
    sink = IdempotentPaymentEventSink(inner, store, clock=lambda: T0)

    with pytest.raises(RuntimeError):
        await sink.handle(_event("evt_1"))
    assert store.failed_events() != []

    # Ops (or an automatic retry) identifies the failed event and redelivers it; the handler now
    # succeeds because the underlying issue was transient.
    inner.fail_on.clear()
    await sink.handle(_event("evt_1"))

    assert store.failed_events() == []
    record = store.get("evt_1")
    assert record is not None and record.status == "done"
    assert inner.events == [_event("evt_1")]

    # A third delivery of the now-done event is still a no-op.
    await sink.handle(_event("evt_1"))
    assert inner.events == [_event("evt_1")]


async def test_already_processed_and_mark_processing_contract() -> None:
    store = InMemoryIdempotencyStore()
    assert await store.already_processed("evt_1") is False

    await store.mark_processing("evt_1", "payment_intent.amount_capturable_updated", T0)
    assert await store.already_processed("evt_1") is False  # processing, not done

    await store.mark_done("evt_1", T0)
    assert await store.already_processed("evt_1") is True

    # mark_processing on an already-done event does not regress its status.
    await store.mark_processing("evt_1", "payment_intent.amount_capturable_updated", T0)
    record = store.get("evt_1")
    assert record is not None and record.status == "done"
