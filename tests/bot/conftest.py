from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from regulars.bot.clock import FrozenClock
from regulars.bot.deposits import FakeDepositCollector
from regulars.bot.onboarding import OnboardingState
from regulars.bot.plan import Slot
from regulars.bot.runtime import BotRuntime, build_runtime
from regulars.bot.store import InMemoryBotStore, Member
from regulars.bot.transport import Attachment, FakeTransport, InboundMessage

T0 = datetime(2026, 9, 14, 17, 0, tzinfo=UTC)

NAMES = ["Priya", "Dev", "Anika", "Rohan", "Meera", "Sam"]


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(T0)


@pytest.fixture
def transport() -> FakeTransport:
    return FakeTransport()


@pytest.fixture
def store(clock: FrozenClock) -> InMemoryBotStore:
    return InMemoryBotStore(now=clock.now)


@pytest.fixture
def deposits() -> FakeDepositCollector:
    return FakeDepositCollector()


@pytest.fixture
def runtime(
    transport: FakeTransport,
    store: InMemoryBotStore,
    deposits: FakeDepositCollector,
    clock: FrozenClock,
) -> BotRuntime:
    return build_runtime(transport, store, deposits, clock)


@pytest.fixture
def members(store: InMemoryBotStore) -> list[Member]:
    """Six people who already finished onboarding (assembly only groups onboarded users)."""
    out: list[Member] = []
    for i, name in enumerate(NAMES, start=1):
        m = Member(user_id=f"u{i}", handle=f"+1555000000{i}", name=name)
        store.bind_handle(m.handle, m.user_id)
        store.save_onboarding(OnboardingState(user_id=m.user_id, stage="done"))
        out.append(m)
    return out


@pytest.fixture
def hooks(members: list[Member]) -> dict[str, str]:
    return {
        m.user_id: f"{m.name} moved here eighteen months ago and still cooks for eight"
        for m in members
    }


@pytest.fixture
def slots() -> list[Slot]:
    return [
        Slot(id="s1", venue="Rooh", starts_at=T0 + timedelta(days=3, hours=2, minutes=30)),
        Slot(id="s2", venue="Besharam", starts_at=T0 + timedelta(days=4, hours=2)),
        Slot(id="s3", venue="Ettan", starts_at=T0 + timedelta(days=5, hours=2)),
    ]


def dm(handle: str, body: str, at: datetime = T0, *, audio: bool = False) -> InboundMessage:
    attachments = (Attachment("audio", f"https://bb.local/att/{handle}/{body or 'voice'}"),)
    return InboundMessage(
        chat_id=f"iMessage;-;{handle}",
        sender_handle=handle,
        body=body,
        sent_at_ms=int(at.timestamp() * 1000),
        message_id=f"m-{handle}-{int(at.timestamp())}-{body[:8]}",
        attachments=attachments if audio else (),
    )


def group_msg(
    chat_id: str, handle: str, body: str, at: datetime, *, tapback: str | None = None
) -> InboundMessage:
    return InboundMessage(
        chat_id=chat_id,
        sender_handle=handle,
        body=tapback or body,
        sent_at_ms=int(at.timestamp() * 1000),
        message_id=f"g-{handle}-{int(at.timestamp())}",
        tapback=tapback,
    )
