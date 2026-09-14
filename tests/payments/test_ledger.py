from dataclasses import dataclass, field

import pytest

from regulars.payments.ledger import (
    InMemoryLedgerStore,
    InsufficientCreditError,
    Ledger,
    LedgerEntry,
    Settlement,
)

DEPOSIT = 2000
USERS = ["u1", "u2", "u3", "u4", "u5", "u6"]


@dataclass
class RecordingGateway:
    captures: list[tuple[str, int | None]] = field(default_factory=list)
    releases: list[str] = field(default_factory=list)

    async def capture(self, intent_id: str, amount_cents: int | None = None) -> None:
        self.captures.append((intent_id, amount_cents))

    async def release(self, intent_id: str) -> None:
        self.releases.append(intent_id)


@pytest.fixture
def store() -> InMemoryLedgerStore:
    return InMemoryLedgerStore()


@pytest.fixture
def gateway() -> RecordingGateway:
    return RecordingGateway()


@pytest.fixture
def ledger(store: InMemoryLedgerStore, gateway: RecordingGateway) -> Ledger:
    return Ledger(store, gateway)


async def six_paid(ledger: Ledger, plan: str = "p1") -> None:
    for u in USERS:
        await ledger.hold(plan, u, f"pi_{u}", DEPOSIT)


async def test_hold_writes_deposit_hold_row(ledger: Ledger, store: InMemoryLedgerStore) -> None:
    hold = await ledger.hold("p1", "u1", "pi_u1", DEPOSIT)
    assert hold.state == "held" and hold.total_cents == DEPOSIT
    [entry] = await store.entries(user_id="u1")
    assert (entry.kind, entry.amount_cents, entry.plan_id) == ("deposit_hold", DEPOSIT, "p1")
    assert await store.holds("p1") == [hold]


async def test_hold_validation(ledger: Ledger) -> None:
    with pytest.raises(ValueError):
        await ledger.hold("p1", "u1", None, 0)
    with pytest.raises(ValueError):
        await ledger.hold("p1", "u1", None, 500)
    with pytest.raises(ValueError):
        await ledger.hold("p1", "u1", "pi", 0, credit_cents=500)


async def test_settle_six_paid_five_attended(
    ledger: Ledger, store: InMemoryLedgerStore, gateway: RecordingGateway
) -> None:
    await six_paid(ledger)
    attended = USERS[:5]
    s = await ledger.settle_plan("p1", attended=attended, paid=USERS, platform_cut_bps=1000)

    assert [h.user_id for h in s.released] == attended
    assert [h.user_id for h in s.forfeited] == ["u6"]
    assert s.forfeited_cents == 2000
    assert s.credits == {u: 360 for u in attended}
    assert s.platform_cents == 200
    assert s.platform_cents + sum(s.credits.values()) == s.forfeited_cents

    assert gateway.releases == [f"pi_{u}" for u in attended]
    assert gateway.captures == [("pi_u6", 2000)]

    forfeits = [e for e in await store.entries(plan_id="p1") if e.kind == "forfeit"]
    assert [(e.user_id, e.amount_cents) for e in forfeits] == [("u6", 2000)]
    for u in attended:
        assert await ledger.available_credit(u) == 360
    assert await ledger.available_credit("u6") == 0
    states = {h.user_id: h.state for h in await store.holds("p1")}
    assert states == {**{u: "released" for u in attended}, "u6": "captured"}


async def test_settle_is_idempotent(
    ledger: Ledger, store: InMemoryLedgerStore, gateway: RecordingGateway
) -> None:
    await six_paid(ledger)
    first = await ledger.settle_plan("p1", attended=USERS[:5], paid=USERS)
    rows = len(await store.entries())
    second = await ledger.settle_plan("p1", attended=USERS[:5], paid=USERS)
    assert isinstance(second, Settlement)
    assert second == first
    assert len(await store.entries()) == rows
    assert len(gateway.captures) == 1 and len(gateway.releases) == 5
    assert await ledger.available_credit("u1") == 360


async def test_settle_resumes_after_partial_credit_write(
    ledger: Ledger, store: InMemoryLedgerStore
) -> None:
    await six_paid(ledger)
    # simulate a crash after u1's credit row landed but before the others
    await store.add_entry(
        LedgerEntry(user_id="u1", kind="redistributed_credit", amount_cents=360, plan_id="p1")
    )
    s = await ledger.settle_plan("p1", attended=USERS[:5], paid=USERS)
    assert s.credits == {u: 360 for u in USERS[:5]} and s.platform_cents == 200
    credit_rows = [e for e in await store.entries(plan_id="p1") if e.kind == "redistributed_credit"]
    assert sorted(e.user_id for e in credit_rows) == USERS[:5]


async def test_settle_remainder_goes_to_platform(ledger: Ledger) -> None:
    await six_paid(ledger)
    # two no-shows: pool 4000, cut 400, 3600 / 4 attendees = 900 exactly
    s = await ledger.settle_plan("p1", attended=USERS[:4], paid=USERS)
    assert s.credits == {u: 900 for u in USERS[:4]} and s.platform_cents == 400
    # 25% cut: 3000 / 4 = 750
    await six_paid(ledger, "p2")
    s2 = await ledger.settle_plan("p2", attended=USERS[:4], paid=USERS, platform_cut_bps=2500)
    assert s2.credits == {u: 750 for u in USERS[:4]} and s2.platform_cents == 1000
    # odd split: 3 no-shows -> pool 6000, cut 600, 5400 / 3 = 1800; with 7 attendees remainder
    for u in ["u7", "u8", "u9", "u10"]:
        await ledger.hold("p3", u, f"pi_{u}", DEPOSIT)
    await six_paid(ledger, "p3")
    everyone = USERS + ["u7", "u8", "u9", "u10"]
    s3 = await ledger.settle_plan("p3", attended=everyone[:7], paid=everyone)
    assert set(s3.credits.values()) == {771}  # 5400 // 7
    assert s3.platform_cents == 6000 - 771 * 7


async def test_settle_ignores_unpaid_and_handles_no_attendees(
    ledger: Ledger, gateway: RecordingGateway, store: InMemoryLedgerStore
) -> None:
    await six_paid(ledger)
    s = await ledger.settle_plan("p1", attended=[], paid=["u1", "u2"])
    assert [h.user_id for h in s.forfeited] == ["u1", "u2"]
    assert s.credits == {} and s.platform_cents == 4000
    assert gateway.releases == []
    assert {h.state for h in await store.holds("p1") if h.user_id not in {"u1", "u2"}} == {"held"}


async def test_apply_credit_reduces_next_hold_and_writes_applied(
    ledger: Ledger, store: InMemoryLedgerStore
) -> None:
    await six_paid(ledger)
    await ledger.settle_plan("p1", attended=USERS[:5], paid=USERS)
    assert await ledger.available_credit("u1") == 360

    applied = await ledger.apply_credit("u1", "p2", 360)
    assert applied == 360
    await ledger.hold("p2", "u1", "pi_p2_u1", DEPOSIT - applied, credit_cents=applied)
    [hold] = await store.holds("p2")
    assert hold.amount_cents == 1640 and hold.credit_cents == 360 and hold.total_cents == DEPOSIT
    assert await ledger.available_credit("u1") == 0
    applied_rows = [e for e in await store.entries(user_id="u1") if e.kind == "applied"]
    assert [(e.plan_id, e.amount_cents) for e in applied_rows] == [("p2", 360)]

    with pytest.raises(InsufficientCreditError):
        await ledger.apply_credit("u1", "p3", 1)
    with pytest.raises(ValueError):
        await ledger.apply_credit("u2", "p3", 0)


async def test_release_returns_applied_credit_and_forfeit_includes_it(
    ledger: Ledger, store: InMemoryLedgerStore, gateway: RecordingGateway
) -> None:
    await six_paid(ledger)
    await ledger.settle_plan("p1", attended=USERS[:5], paid=USERS)
    for u in ["u1", "u2"]:
        await ledger.apply_credit(u, "p2", 360)
        await ledger.hold("p2", u, f"pi2_{u}", DEPOSIT - 360, credit_cents=360)
    await ledger.hold("p2", "u3", "pi2_u3", DEPOSIT)

    s = await ledger.settle_plan("p2", attended=["u1", "u3"], paid=["u1", "u2", "u3"])
    # u1 attended with credit applied: hold released, credit restored
    assert gateway.releases[-2:] == ["pi2_u1", "pi2_u3"]
    assert await ledger.available_credit("u1") == 360 + s.credits["u1"]
    # u2 no-show with credit applied: Stripe captures 1640, forfeit row is the full 2000
    assert gateway.captures[-1] == ("pi2_u2", 1640)
    assert s.forfeited_cents == 2000
    assert s.credits == {"u1": 900, "u3": 900} and s.platform_cents == 200
    assert await ledger.available_credit("u2") == 0
    reversal = [e for e in await store.entries(user_id="u1", plan_id="p2") if e.kind == "applied"]
    assert sorted(e.amount_cents for e in reversal) == [-360, 360]


async def test_credit_only_hold_is_forfeitable_without_stripe(
    ledger: Ledger, gateway: RecordingGateway
) -> None:
    await six_paid(ledger)
    await ledger.settle_plan("p1", attended=USERS[:5], paid=USERS)
    await ledger.apply_credit("u1", "p2", 360)
    await ledger.hold("p2", "u1", None, 0, credit_cents=360)
    await ledger.hold("p2", "u2", "pi2_u2", DEPOSIT)
    captures_before = len(gateway.captures)
    s = await ledger.settle_plan("p2", attended=["u2"], paid=["u1", "u2"])
    assert len(gateway.captures) == captures_before  # nothing to capture for a credit-only hold
    assert s.forfeited_cents == 360 and s.credits == {"u2": 324} and s.platform_cents == 36


async def test_forfeit_and_release_conflicts(ledger: Ledger) -> None:
    await ledger.hold("p1", "u1", "pi_u1", DEPOSIT)
    await ledger.forfeit("p1", "u1")
    with pytest.raises(ValueError):
        await ledger.release("p1", "u1")
    with pytest.raises(LookupError):
        await ledger.forfeit("p1", "nobody")
    with pytest.raises(ValueError):
        await ledger.settle_plan("p1", attended=[], paid=[], platform_cut_bps=10_001)


async def test_ledger_without_gateway_only_records(store: InMemoryLedgerStore) -> None:
    ledger = Ledger(store)
    await ledger.hold("p1", "u1", "pi_u1", DEPOSIT)
    await ledger.hold("p1", "u2", "pi_u2", DEPOSIT)
    s = await ledger.settle_plan("p1", attended=["u1"], paid=["u1", "u2"])
    assert [h.intent_id for h in s.released] == ["pi_u1"]
    assert [h.intent_id for h in s.forfeited] == ["pi_u2"]
    assert s.credits == {"u1": 1800}
