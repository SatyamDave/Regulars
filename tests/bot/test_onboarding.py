from __future__ import annotations

from regulars.bot import voice
from regulars.bot.events import OnboardingEvent
from regulars.bot.runtime import BotRuntime
from regulars.bot.store import InMemoryBotStore
from regulars.bot.transport import FakeTransport
from tests.bot.conftest import dm

HANDLE = "+14155550100"


async def test_stranger_to_interview_complete(
    runtime: BotRuntime, transport: FakeTransport, store: InMemoryBotStore
) -> None:
    await runtime.process(dm(HANDLE, "hi"))
    first = transport.dms_to(HANDLE)
    assert first == [voice.WELCOME, voice.WELCOME_HOW, voice.INTERVIEW_PROMPTS[0]]
    user = store.user_for_handle(HANDLE)
    assert user is not None
    state = store.get_onboarding(user)
    assert state is not None and state.stage == "prompt_1"

    await runtime.process(dm(HANDLE, "", audio=True))
    assert transport.dms_to(HANDLE)[-1] == voice.INTERVIEW_PROMPTS[1]

    await runtime.process(dm(HANDLE, "I stopped seeing Raj because every plan was his plan."))
    assert transport.dms_to(HANDLE)[-1] == voice.INTERVIEW_PROMPTS[2]
    assert runtime.events == []

    await runtime.process(dm(HANDLE, "", audio=True))
    assert transport.dms_to(HANDLE)[-1] == voice.INTERVIEW_DONE

    assert len(runtime.events) == 1
    event = runtime.events[0]
    assert isinstance(event, OnboardingEvent)
    assert event.kind == "interview_complete"
    assert event.user_id == user
    assert [a.prompt_index for a in event.answers] == [0, 1, 2]
    assert [a.typed for a in event.answers] == [False, True, False]
    assert event.answers[0].audio_url is not None and event.answers[0].audio_url.startswith("http")
    assert event.answers[1].text.startswith("I stopped seeing Raj")
    assert event.answers[1].audio_url is None

    final = store.get_onboarding(user)
    assert final is not None and final.done
    # every reply went through send_message -> an audit row each
    assert [a.decision_type for a in store.audit] == ["send_message"] * len(
        transport.dms_to(HANDLE)
    )


async def test_empty_message_nudges_without_advancing(
    runtime: BotRuntime, transport: FakeTransport, store: InMemoryBotStore
) -> None:
    await runtime.process(dm(HANDLE, "hello"))
    await runtime.process(dm(HANDLE, "   "))
    assert transport.dms_to(HANDLE)[-1] == voice.INTERVIEW_NUDGE
    user = store.user_for_handle(HANDLE)
    assert user is not None
    state = store.get_onboarding(user)
    assert state is not None and state.stage == "prompt_1" and state.answers == ()


async def test_done_user_gets_fallback_not_interview(
    runtime: BotRuntime, transport: FakeTransport
) -> None:
    for body in ("hi", "one", "two", "three"):
        await runtime.process(dm(HANDLE, body))
    await runtime.process(dm(HANDLE, "are you there"))
    assert transport.dms_to(HANDLE)[-1] == voice.DM_FALLBACK
