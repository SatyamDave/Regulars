"""Onboarding: the three-prompt voice interview (product-plan §3), as a pure state machine.

``welcome -> prompt_1 -> prompt_2 -> prompt_3 -> done``. Each step takes the stored state and
one inbound message and returns the new state, the replies to send, and any events. Audio
attachments are the expected answer; a text answer is accepted and flagged ``typed=True``.
Completion emits ``OnboardingEvent(kind="interview_complete")`` for ``regulars.persona``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from regulars.bot import voice
from regulars.bot.events import InterviewAnswer, OnboardingEvent
from regulars.bot.transport import InboundMessage

Stage = Literal["welcome", "prompt_1", "prompt_2", "prompt_3", "done"]
STAGES: tuple[Stage, ...] = ("welcome", "prompt_1", "prompt_2", "prompt_3", "done")
_PROMPT_STAGES: tuple[Stage, ...] = ("prompt_1", "prompt_2", "prompt_3")


class OnboardingState(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    stage: Stage = "welcome"
    answers: tuple[InterviewAnswer, ...] = ()

    @property
    def done(self) -> bool:
        return self.stage == "done"


@dataclass(frozen=True)
class OnboardingStep:
    state: OnboardingState
    replies: tuple[str, ...]
    events: tuple[OnboardingEvent, ...] = ()


class Onboarding:
    def start(self, user_id: str) -> OnboardingStep:
        """First contact: welcome, how it works, and the first prompt."""
        state = OnboardingState(user_id=user_id, stage="prompt_1")
        return OnboardingStep(state, (voice.WELCOME, voice.WELCOME_HOW, voice.INTERVIEW_PROMPTS[0]))

    def step(self, state: OnboardingState, inbound: InboundMessage) -> OnboardingStep:
        if state.stage == "welcome":
            return self.start(state.user_id)
        if state.stage == "done":
            return OnboardingStep(state, ())

        index = _PROMPT_STAGES.index(state.stage)
        answer = answer_from(inbound, index)
        if answer is None:
            return OnboardingStep(state, (voice.INTERVIEW_NUDGE,))

        answers = (*state.answers, answer)
        if index + 1 < len(_PROMPT_STAGES):
            next_stage = _PROMPT_STAGES[index + 1]
            ack = voice.INTERVIEW_ACK[index % len(voice.INTERVIEW_ACK)]
            return OnboardingStep(
                state.model_copy(update={"stage": next_stage, "answers": answers}),
                (ack, voice.INTERVIEW_PROMPTS[index + 1]),
            )
        done = state.model_copy(update={"stage": "done", "answers": answers})
        event = OnboardingEvent(kind="interview_complete", user_id=state.user_id, answers=answers)
        return OnboardingStep(done, (voice.INTERVIEW_DONE,), (event,))


def answer_from(inbound: InboundMessage, prompt_index: int) -> InterviewAnswer | None:
    """Audio wins; text is the flagged fallback; nothing usable returns ``None``."""
    audio = next((a for a in inbound.attachments if a.kind == "audio"), None)
    if audio is not None:
        return InterviewAnswer(
            prompt_index=prompt_index, text=inbound.body, audio_url=audio.url, typed=False
        )
    if inbound.body.strip():
        return InterviewAnswer(prompt_index=prompt_index, text=inbound.body.strip(), typed=True)
    return None
