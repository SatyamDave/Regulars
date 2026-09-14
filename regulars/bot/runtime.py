"""BotRuntime: the loop. ``process`` handles one inbound message; ``tick`` advances time-driven
flows. Events for other subpackages accumulate on ``events`` (and go to ``on_event`` if given).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from regulars.bot.clock import Clock
from regulars.bot.deposits import DepositCollector
from regulars.bot.events import BackfillRequest, BotEvent
from regulars.bot.flows import GroupOpener, LockFlow, MorningAfter
from regulars.bot.onboarding import Onboarding
from regulars.bot.plan import Plan
from regulars.bot.router import Emit, LogSignal, OutboundAction, Router, SendMessage
from regulars.bot.store import BotStore
from regulars.bot.tools import BotTools
from regulars.bot.transport import InboundMessage, Transport


@dataclass
class Flows:
    opener: GroupOpener
    lock: LockFlow
    morning_after: MorningAfter


class BotRuntime:
    def __init__(
        self,
        tools: BotTools,
        router: Router,
        flows: Flows,
        store: BotStore,
        clock: Clock,
        *,
        on_event: Callable[[BotEvent], None] | None = None,
    ) -> None:
        self.tools = tools
        self.router = router
        self.flows = flows
        self.store = store
        self.clock = clock
        self.events: list[BotEvent] = []
        self._on_event = on_event

    async def process(self, inbound: InboundMessage) -> list[OutboundAction]:
        actions = self.router.handle(inbound)
        for action in actions:
            await self._execute(action)
        return actions

    async def tick(self, now: datetime | None = None) -> list[BackfillRequest]:
        backfills = await self.flows.lock.tick(now or self.clock.now())
        for b in backfills:
            self._emit(b)
        return backfills

    async def payment_received(self, plan_id: str, user_id: str) -> Plan:
        """Entry point for the payments webhook. Locks on the next ``tick``."""
        return await self.flows.lock.record_payment(plan_id, user_id)

    async def _execute(self, action: OutboundAction) -> None:
        if isinstance(action, SendMessage):
            await self.tools.send_message(action.group_id, action.body, to_user=action.to_user)
        elif isinstance(action, LogSignal):
            await self.tools.log_signal(
                action.user_id, action.signal_type, action.value, group_id=action.group_id
            )
        elif isinstance(action, Emit):
            self._emit(action.event)

    def _emit(self, event: BotEvent) -> None:
        self.events.append(event)
        if self._on_event is not None:
            self._on_event(event)


def build_runtime(
    transport: Transport,
    store: BotStore,
    deposits: DepositCollector,
    clock: Clock,
    *,
    on_event: Callable[[BotEvent], None] | None = None,
) -> BotRuntime:
    """Wire the default runtime. Every seam is a parameter; nothing here reaches the network."""
    tools = BotTools(transport, store, deposits, clock)
    flows = Flows(
        opener=GroupOpener(tools, transport, store, clock),
        lock=LockFlow(tools, store, clock),
        morning_after=MorningAfter(tools, store, clock),
    )
    router = Router(store, Onboarding(), clock)
    return BotRuntime(tools, router, flows, store, clock, on_event=on_event)
