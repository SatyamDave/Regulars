from collections.abc import Awaitable, Callable

import pytest
import stripe as stripe_sdk

from regulars.payments.collector import DroppedEvent, PaidEvent, handle_payment_event
from regulars.payments.stripe_deposits import StripeDeposits, WebhookSignatureError
from tests.payments.fakes import FakeStripeClient, event_json

SECRET = "whsec_test"


@pytest.fixture
def client() -> FakeStripeClient:
    return FakeStripeClient()


@pytest.fixture
def deposits(client: FakeStripeClient) -> StripeDeposits:
    return StripeDeposits("sk_test_fake", SECRET, client=client)


async def test_create_hold_is_manual_capture_with_metadata(
    deposits: StripeDeposits, client: FakeStripeClient
) -> None:
    hold = await deposits.create_hold("plan-1", "user-a", 2000, customer_id="cus_1")
    assert hold.intent_id == "pi_1"
    assert hold.client_secret == "pi_1_secret"
    assert hold.status == "requires_payment_method"
    [params] = client.v1.payment_intents.created
    assert params == {
        "amount": 2000,
        "currency": "usd",
        "capture_method": "manual",
        "metadata": {"plan_id": "plan-1", "user_id": "user-a"},
        "customer": "cus_1",
    }


async def test_create_hold_without_customer_omits_key(
    deposits: StripeDeposits, client: FakeStripeClient
) -> None:
    await deposits.create_hold("plan-1", "user-a", 2000)
    assert "customer" not in client.v1.payment_intents.created[0]
    with pytest.raises(ValueError):
        await deposits.create_hold("plan-1", "user-a", 0)


async def test_capture_and_release(deposits: StripeDeposits, client: FakeStripeClient) -> None:
    full = await deposits.capture("pi_9")
    assert full.status == "succeeded"
    partial = await deposits.capture("pi_8", 1500)
    assert partial.intent_id == "pi_8"
    assert client.v1.payment_intents.captured == [
        ("pi_9", None),
        ("pi_8", {"amount_to_capture": 1500}),
    ]
    released = await deposits.release("pi_7")
    assert released.status == "canceled"
    assert client.v1.payment_intents.canceled == ["pi_7"]


def test_verify_webhook_good_signature(deposits: StripeDeposits) -> None:
    payload = event_json("payment_intent.amount_capturable_updated", "pi_5", "plan-1", "user-a")
    event = deposits.verify_webhook(payload, f"sig:{SECRET}")
    assert event.id == "evt_1"
    assert event.type == "payment_intent.amount_capturable_updated"
    assert event.object["id"] == "pi_5"
    assert event.metadata == {"plan_id": "plan-1", "user_id": "user-a"}
    assert handle_payment_event(event) == PaidEvent("plan-1", "user-a", "pi_5")


def test_verify_webhook_bad_signature(deposits: StripeDeposits) -> None:
    payload = event_json("payment_intent.canceled", "pi_5", "plan-1", "user-a")
    with pytest.raises(WebhookSignatureError):
        deposits.verify_webhook(payload, "sig:wrong")
    with pytest.raises(WebhookSignatureError):
        deposits.verify_webhook(payload, None)
    with pytest.raises(WebhookSignatureError):
        deposits.verify_webhook(b"not json", f"sig:{SECRET}")


def test_canceled_event_maps_to_dropped(deposits: StripeDeposits) -> None:
    payload = event_json("payment_intent.canceled", "pi_5", "plan-1", "user-a")
    event = deposits.verify_webhook(payload, f"sig:{SECRET}")
    assert handle_payment_event(event) == DroppedEvent("plan-1", "user-a", "pi_5")


# --- outbound retries (regulars.core.retry) ------------------------------------------------


async def test_create_hold_retries_transient_errors(
    deposits: StripeDeposits, client: FakeStripeClient
) -> None:
    calls = 0
    real_create = client.v1.payment_intents.create_async

    async def flaky_create(params: object) -> object:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise stripe_sdk.RateLimitError("rate limited")
        return await real_create(params)

    client.v1.payment_intents.create_async = flaky_create  # type: ignore[method-assign]
    slept: list[float] = []
    deposits.sleep = _recording_sleep(slept)

    hold = await deposits.create_hold("plan-1", "user-a", 2000)
    assert hold.intent_id == "pi_1"
    assert calls == 3
    assert len(slept) == 2


async def test_create_hold_gives_up_after_max_retries(
    deposits: StripeDeposits, client: FakeStripeClient
) -> None:
    async def always_rate_limited(params: object) -> object:
        raise stripe_sdk.RateLimitError("rate limited")

    client.v1.payment_intents.create_async = always_rate_limited  # type: ignore[method-assign]
    deposits.sleep = _recording_sleep([])
    with pytest.raises(stripe_sdk.RateLimitError):
        await deposits.create_hold("plan-1", "user-a", 2000)


async def test_capture_does_not_retry_non_transient_errors(
    deposits: StripeDeposits, client: FakeStripeClient
) -> None:
    async def bad_request(intent: str, params: object = None) -> object:
        raise stripe_sdk.InvalidRequestError("bad request", param=None)

    client.v1.payment_intents.capture_async = bad_request  # type: ignore[method-assign]
    calls: list[float] = []
    deposits.sleep = _recording_sleep(calls)
    with pytest.raises(stripe_sdk.InvalidRequestError):
        await deposits.capture("pi_9")
    assert calls == []


def _recording_sleep(slept: list[float]) -> Callable[[float], Awaitable[None]]:
    async def sleep(delay: float) -> None:
        slept.append(delay)

    return sleep
