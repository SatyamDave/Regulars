import pytest

from regulars.payments.collector import (
    DepositCollector,
    DepositRequest,
    DroppedEvent,
    FakeDepositCollector,
    PaidEvent,
    StripeDepositCollector,
    handle_payment_event,
)
from regulars.payments.ledger import InMemoryLedgerStore, Ledger
from regulars.payments.stripe_deposits import StripeDeposits, StripeEvent
from tests.payments.fakes import FakeStripeClient


@pytest.fixture
def client() -> FakeStripeClient:
    return FakeStripeClient()


@pytest.fixture
def store() -> InMemoryLedgerStore:
    return InMemoryLedgerStore()


@pytest.fixture
def collector(client: FakeStripeClient, store: InMemoryLedgerStore) -> StripeDepositCollector:
    deposits = StripeDeposits("sk_test_fake", "whsec", client=client)
    return StripeDepositCollector(deposits, Ledger(store, deposits))


async def test_collect_without_credit_holds_full_amount(
    collector: StripeDepositCollector, client: FakeStripeClient, store: InMemoryLedgerStore
) -> None:
    req = await collector.collect("p1", "u1", 2000)
    assert req == DepositRequest("pi_1", "pi_1_secret", 2000)
    [params] = client.v1.payment_intents.created
    assert params["amount"] == 2000 and params["capture_method"] == "manual"
    [hold] = await store.holds("p1")
    assert (hold.intent_id, hold.amount_cents, hold.credit_cents) == ("pi_1", 2000, 0)


async def test_collect_applies_credit_first(
    collector: StripeDepositCollector, client: FakeStripeClient, store: InMemoryLedgerStore
) -> None:
    # earn 360 of credit on plan p1 (6 paid, 5 attended), then collect for p2
    for i in range(1, 7):
        await collector.collect("p1", f"u{i}", 2000)
    ledger = Ledger(store)
    await ledger.settle_plan(
        "p1", attended=[f"u{i}" for i in range(1, 6)], paid=[f"u{i}" for i in range(1, 7)]
    )
    assert await ledger.available_credit("u1") == 360

    req = await collector.collect("p2", "u1", 2000)
    assert req == DepositRequest("pi_7", "pi_7_secret", 1640)
    assert client.v1.payment_intents.created[-1]["amount"] == 1640
    assert await ledger.available_credit("u1") == 0
    applied = [e for e in await store.entries(user_id="u1", plan_id="p2") if e.kind == "applied"]
    assert [e.amount_cents for e in applied] == [360]
    [hold] = await store.holds("p2")
    assert hold.amount_cents == 1640 and hold.credit_cents == 360


async def test_collect_fully_covered_by_credit_skips_stripe(
    collector: StripeDepositCollector, client: FakeStripeClient, store: InMemoryLedgerStore
) -> None:
    for i in range(1, 7):
        await collector.collect("p1", f"u{i}", 2000)
    ledger = Ledger(store)
    await ledger.settle_plan("p1", attended=["u1"], paid=[f"u{i}" for i in range(1, 7)])
    assert await ledger.available_credit("u1") == 9000
    created = len(client.v1.payment_intents.created)

    req = await collector.collect("p2", "u1", 2000)
    assert req == DepositRequest(None, None, 0)
    assert len(client.v1.payment_intents.created) == created
    assert await ledger.available_credit("u1") == 7000
    [hold] = await store.holds("p2")
    assert hold.intent_id is None and hold.credit_cents == 2000


async def test_collect_rejects_non_positive(collector: StripeDepositCollector) -> None:
    with pytest.raises(ValueError):
        await collector.collect("p1", "u1", 0)


async def test_fake_collector_matches_protocol() -> None:
    fake: DepositCollector = FakeDepositCollector()
    req = await fake.collect("p1", "u1", 2000)
    assert req == DepositRequest("pi_fake_1", "pi_fake_1_secret", 2000)
    assert isinstance(fake, FakeDepositCollector) and fake.requests == [("p1", "u1", 2000)]


def _event(kind: str, **obj: object) -> StripeEvent:
    return StripeEvent(id="evt", type=kind, object=dict(obj))


def test_handle_payment_event() -> None:
    meta = {"plan_id": "p1", "user_id": "u1"}
    paid = _event("payment_intent.amount_capturable_updated", id="pi_1", metadata=meta)
    assert handle_payment_event(paid) == PaidEvent("p1", "u1", "pi_1")
    dropped = _event("payment_intent.canceled", id="pi_1", metadata=meta)
    assert handle_payment_event(dropped) == DroppedEvent("p1", "u1", "pi_1")
    assert (
        handle_payment_event(_event("payment_intent.succeeded", id="pi_1", metadata=meta)) is None
    )
    assert handle_payment_event(_event("payment_intent.canceled", id="pi_1", metadata={})) is None
    assert handle_payment_event(_event("payment_intent.canceled", id="pi_1")) is None
