"""The three-prompt voice interview (product-plan §3 Onboarding).

Ninety seconds per prompt, freeform voice. :class:`InterviewState` is immutable: recording an
answer returns a new state, so the bot can persist it between iMessage turns without worrying
about partial mutation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from regulars.persona.prosody import ProsodyFeatures

INTERVIEW_PROMPTS: tuple[str, str, str] = (
    "Walk me through your last genuinely good weekend.",
    "Tell me about someone you stopped hanging out with, and why.",
    "What are you actually looking for right now?",
)
PROMPT_SECONDS = 90


class InterviewAnswer(BaseModel, frozen=True):
    prompt_index: int = Field(ge=0, lt=len(INTERVIEW_PROMPTS))
    text: str
    prosody: ProsodyFeatures | None = None
    typed: bool = False
    """True when the user typed instead of sending voice; no prosody is expected then."""

    @field_validator("text")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("an interview answer needs some text")
        return v.strip()

    @property
    def prompt(self) -> str:
        return INTERVIEW_PROMPTS[self.prompt_index]


class InterviewState(BaseModel, frozen=True):
    answers: tuple[InterviewAnswer, ...] = ()

    @field_validator("answers")
    @classmethod
    def _in_order(cls, v: tuple[InterviewAnswer, ...]) -> tuple[InterviewAnswer, ...]:
        indices = [a.prompt_index for a in v]
        if indices != list(range(len(v))):
            raise ValueError(f"answers must cover prompts 0..n-1 in order, got {indices}")
        return v

    @property
    def next_prompt_index(self) -> int | None:
        i = len(self.answers)
        return i if i < len(INTERVIEW_PROMPTS) else None

    @property
    def next_prompt(self) -> str | None:
        i = self.next_prompt_index
        return None if i is None else INTERVIEW_PROMPTS[i]

    @property
    def is_complete(self) -> bool:
        return self.next_prompt_index is None

    def record_answer(
        self, text: str, prosody: ProsodyFeatures | None = None, *, typed: bool = False
    ) -> InterviewState:
        """Return a new state with the answer to the current prompt appended."""
        i = self.next_prompt_index
        if i is None:
            raise ValueError("interview is already complete")
        answer = InterviewAnswer(prompt_index=i, text=text, prosody=prosody, typed=typed)
        return InterviewState(answers=(*self.answers, answer))
