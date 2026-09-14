"""Glue the ten subpackages into one running system (technical-architecture.md §0, §9).

This is the one module allowed to import across every ``regulars.*`` subpackage: it is the
integration layer the packages are built to be wired *by*, not one subpackage reaching into
another's internals (the rule in ``CLAUDE.md`` is about the subpackages, not their assembler).

:func:`build_system` constructs the SQL-backed bot store and ledger, a deposit collector (real
Stripe when configured, a fake one otherwise), the bot runtime, and the FastAPI app with sinks
that turn a BlueBubbles webhook into ``runtime.process`` and a Stripe webhook into
``runtime.payment_received``. :class:`System` bundles the result; :meth:`System.from_env` reads
``Settings`` and ``DATABASE_URL`` for a real deployment.

**Seam this module has to paper over:** ``regulars.bot.deposits.DepositCollector`` requires a
non-optional ``intent_id``/``pay_url_or_secret`` (the bot always has something to text the
member), while ``regulars.payments.collector.DepositCollector`` returns ``None`` for both when
credit covers a deposit in full. :class:`_CollectorAdapter` bridges the two.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from sqlalchemy import Engine

from regulars.app import Settings, create_app
from regulars.bot.clock import Clock, SystemClock
from regulars.bot.deposits import DepositRequest as BotDepositRequest
from regulars.bot.runtime import BotRuntime, build_runtime
from regulars.bot.transport import Transport, parse_bluebubbles_webhook
from regulars.db import repo
from regulars.db.bot_store import SqlBotStore
from regulars.db.engine import make_engine
from regulars.db.idempotency_store import SqlIdempotencyStore
from regulars.db.ledger_store import SqlLedgerStore
from regulars.observability.instrumented import bot_tool_hook
from regulars.observability.logging_config import configure_logging
from regulars.observability.middleware import RequestLoggingMiddleware
from regulars.payments.collector import (
    DepositCollector as PaymentsDepositCollector,
)
from regulars.payments.collector import (
    DroppedEvent,
    FakeDepositCollector,
    PaidEvent,
    StripeDepositCollector,
    handle_payment_event,
)
from regulars.payments.fake_stripe import FakeStripeClient
from regulars.payments.idempotency import IdempotentPaymentEventSink, WebhookIdempotencyStore
from regulars.payments.ledger import Ledger
from regulars.payments.stripe_deposits import StripeDeposits, StripeEvent

CREDIT_ONLY_INTENT = "credit_covered"
"""Sentinel ``intent_id`` the bot sees when a deposit was fully covered by credit."""


@dataclass(frozen=True)
class _CollectorAdapter:
    """``regulars.payments`` collector -> ``regulars.bot.deposits.DepositCollector`` protocol."""

    inner: PaymentsDepositCollector

    async def collect(self, plan_id: str, user_id: str, amount_cents: int) -> BotDepositRequest:
        request = await self.inner.collect(plan_id, user_id, amount_cents)
        if request.intent_id is None:
            return BotDepositRequest(CREDIT_ONLY_INTENT, "", 0)
        return BotDepositRequest(
            request.intent_id, request.pay_url_or_secret or "", request.amount_after_credit_cents
        )


@dataclass
class _InboundSink:
    handle_fn: Callable[[dict[str, Any]], Any]

    async def handle(self, payload: dict[str, Any]) -> None:
        await self.handle_fn(payload)


@dataclass
class _PaymentSink:
    handle_fn: Callable[[StripeEvent], Any]

    async def handle(self, event: StripeEvent) -> None:
        await self.handle_fn(event)


@dataclass
class System:
    app: FastAPI
    runtime: BotRuntime
    engine: Engine
    store: SqlBotStore
    ledger: Ledger

    @classmethod
    def from_env(
        cls,
        *,
        transport: Transport,
        stripe: StripeDeposits | None = None,
        clock: Clock | None = None,
        env: Mapping[str, str] | None = None,
        idempotency_store: WebhookIdempotencyStore | None = None,
        observability: bool = True,
    ) -> System:
        """Read ``Settings`` and ``DATABASE_URL`` from the environment and build the system."""
        settings = Settings.from_env(env)
        engine = make_engine()
        return build_system(
            settings,
            engine=engine,
            transport=transport,
            stripe=stripe,
            clock=clock,
            idempotency_store=idempotency_store,
            observability=observability,
        )


def build_system(
    settings: Settings,
    *,
    engine: Engine,
    transport: Transport,
    stripe: StripeDeposits | None = None,
    clock: Clock | None = None,
    idempotency_store: WebhookIdempotencyStore | None = None,
    observability: bool = True,
) -> System:
    """Wire one ``System``: SQL stores, ledger, deposit collector, bot runtime, FastAPI app.

    ``idempotency_store`` guards ``/webhooks/stripe`` against Stripe's at-least-once redelivery
    (see ``regulars.payments.idempotency``); it defaults to a ``SqlIdempotencyStore`` over
    ``engine`` so production wiring is protected without callers having to opt in.

    ``observability=True`` (the default) additionally: configures JSON stdlib logging, adds
    ``RequestLoggingMiddleware`` to the app, and passes a metrics-recording ``on_call`` hook into
    the bot runtime's ``BotTools`` (§9). Pass ``observability=False`` to opt out (e.g. a test that
    configures logging itself) with the rest of the wiring unchanged.
    """
    active_clock = clock or SystemClock()
    store = SqlBotStore(engine)
    ledger_store = SqlLedgerStore(engine)
    ledger = Ledger(ledger_store, gateway=stripe, clock=active_clock.now)

    payments_collector: PaymentsDepositCollector
    if stripe is not None:
        payments_collector = StripeDepositCollector(stripe, ledger)
    else:
        payments_collector = FakeDepositCollector()
    bot_deposits = _CollectorAdapter(payments_collector)

    if observability:
        configure_logging()

    runtime = build_runtime(
        transport,
        store,
        bot_deposits,
        active_clock,
        on_tool_call=bot_tool_hook() if observability else None,
    )

    async def handle_inbound(payload: dict[str, Any]) -> None:
        inbound = parse_bluebubbles_webhook(payload)
        if inbound is None:
            return
        group = store.group_for_chat(inbound.chat_id)
        if group is not None:
            member = group.member_by_handle(inbound.sender_handle)
            if member is not None:
                with engine.begin() as conn:
                    repo.insert_message(
                        conn,
                        group_id=group.group_id,
                        user_id=member.user_id,
                        sent_at_ms=inbound.sent_at_ms,
                        body=inbound.body,
                    )
        await runtime.process(inbound)

    async def handle_payment(event: StripeEvent) -> None:
        fact = handle_payment_event(event)
        if isinstance(fact, PaidEvent):
            await runtime.payment_received(fact.plan_id, fact.user_id)
        elif isinstance(fact, DroppedEvent):
            # The ledger already reflects a dropped hold via Ledger.release/forfeit; the bot's
            # own drop path (LockFlow.tick, 24h after deposits go out) is what removes the
            # member and requests a backfill, so there is nothing further to do here.
            pass

    active_idempotency_store = idempotency_store or SqlIdempotencyStore(engine)
    payment_sink = IdempotentPaymentEventSink(
        inner=_PaymentSink(handle_payment),
        store=active_idempotency_store,
        clock=active_clock.now,
    )
    deposits_for_app = stripe if stripe is not None else _offline_stripe_deposits(settings)
    app = create_app(
        settings,
        deposits=deposits_for_app,
        payment_sink=payment_sink,
        inbound_sink=_InboundSink(handle_inbound),
    )
    if observability:
        app.add_middleware(RequestLoggingMiddleware)
    return System(app=app, runtime=runtime, engine=engine, store=store, ledger=ledger)


def _offline_stripe_deposits(settings: Settings) -> StripeDeposits:
    """A ``StripeDeposits`` whose client never touches the network (no key configured)."""
    return StripeDeposits(
        settings.stripe_api_key.get_secret_value(),
        settings.stripe_webhook_secret.get_secret_value(),
        client=FakeStripeClient(),
    )


def fake_stripe_event(event_type: str, *, plan_id: str, user_id: str, intent_id: str) -> bytes:
    """Build the raw JSON body a ``FakeStripeClient``-backed ``/webhooks/stripe`` accepts."""
    return json.dumps(
        {
            "id": f"evt_{intent_id}",
            "type": event_type,
            "data": {
                "object": {
                    "id": intent_id,
                    "metadata": {"plan_id": plan_id, "user_id": user_id},
                }
            },
        }
    ).encode()
