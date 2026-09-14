"""Webhook delivery idempotency (Stripe's own docs: webhooks are delivered *at-least-once*,
retried on any non-2xx response or a timeout).

Nothing upstream of this module deduplicates by Stripe event id, so a replayed
``payment_intent.amount_capturable_updated`` could double-process a payment (double-write a
``PaidEvent``, double-trigger ``runtime.payment_received``). ``WebhookIdempotencyStore`` records
each event id's processing status; ``IdempotentPaymentEventSink`` wraps any sink satisfying
``regulars.app.PaymentEventSink`` so a redelivered event is a no-op, and a handler failure is
recorded (with its error) rather than silently swallowed, so failed events can be found and
retried later.

``regulars.app.PaymentEventSink`` is not imported here (only its shape, via ``_PaymentEventSink``
below): ``regulars.core`` is the only cross-package import surface, and ``regulars.app`` is not
core. Structural typing (``Protocol``) makes the import unnecessary — anything satisfying
``PaymentEventSink`` already satisfies ``_PaymentEventSink``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal, Protocol

from regulars.payments.stripe_deposits import StripeEvent

WebhookStatus = Literal["processing", "done", "failed"]


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ProcessedWebhookEvent:
    """One ``processed_webhook_events`` row as the payments code sees it."""

    event_id: str
    event_type: str
    received_at: datetime
    status: WebhookStatus
    processed_at: datetime | None = None
    error: str | None = None


class WebhookIdempotencyStore(Protocol):
    async def already_processed(self, event_id: str) -> bool:
        """True once ``event_id`` has completed successfully; a replay should be a no-op."""
        ...

    async def mark_processing(self, event_id: str, event_type: str, now: datetime) -> None:
        """Record that ``event_id`` is being handled. A no-op if already ``done``."""
        ...

    async def mark_done(self, event_id: str, now: datetime) -> None: ...

    async def mark_failed(self, event_id: str, now: datetime, error: str) -> None: ...


class InMemoryIdempotencyStore:
    """For tests and the fake/offline wiring path. Not shared across processes."""

    def __init__(self) -> None:
        self._events: dict[str, ProcessedWebhookEvent] = {}

    async def already_processed(self, event_id: str) -> bool:
        existing = self._events.get(event_id)
        return existing is not None and existing.status == "done"

    async def mark_processing(self, event_id: str, event_type: str, now: datetime) -> None:
        existing = self._events.get(event_id)
        if existing is not None and existing.status == "done":
            return
        self._events[event_id] = ProcessedWebhookEvent(
            event_id=event_id, event_type=event_type, received_at=now, status="processing"
        )

    async def mark_done(self, event_id: str, now: datetime) -> None:
        self._events[event_id] = replace(
            self._events[event_id], status="done", processed_at=now, error=None
        )

    async def mark_failed(self, event_id: str, now: datetime, error: str) -> None:
        self._events[event_id] = replace(
            self._events[event_id], status="failed", processed_at=now, error=error
        )

    def get(self, event_id: str) -> ProcessedWebhookEvent | None:
        return self._events.get(event_id)

    def failed_events(self) -> list[ProcessedWebhookEvent]:
        """Events currently ``failed``: candidates a retry job would pick back up."""
        return [e for e in self._events.values() if e.status == "failed"]


class _PaymentEventSink(Protocol):
    """Structurally identical to ``regulars.app.PaymentEventSink`` (see module docstring)."""

    async def handle(self, event: StripeEvent) -> None: ...


@dataclass
class IdempotentPaymentEventSink:
    """Wrap a ``PaymentEventSink`` so a Stripe webhook redelivery is a no-op.

    ``handle`` checks ``event.id`` against ``store`` before delegating to ``inner``: an event
    already marked ``done`` returns immediately without calling ``inner`` again. Otherwise the
    event is marked ``processing``, ``inner.handle`` runs, and the outcome is recorded as ``done``
    or ``failed`` (with the exception's message). A handler exception is re-raised after being
    recorded, so the caller's existing "exceptions propagate as 500s so Stripe retries" behavior
    (``regulars/app.py``) is unchanged — the retry is then a no-op because of the ``store`` check.
    """

    inner: _PaymentEventSink
    store: WebhookIdempotencyStore
    clock: Callable[[], datetime] = _now

    async def handle(self, event: StripeEvent) -> None:
        if await self.store.already_processed(event.id):
            return
        await self.store.mark_processing(event.id, event.type, self.clock())
        try:
            await self.inner.handle(event)
        except Exception as exc:
            await self.store.mark_failed(event.id, self.clock(), str(exc))
            raise
        await self.store.mark_done(event.id, self.clock())
