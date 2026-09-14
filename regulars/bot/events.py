"""Events the bot emits for other subpackages, and the small records they carry.

Nothing here imports from another subpackage. Consumers (persona, assembly) read these off
``BotRuntime.events``; the bot never imports them back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class InterviewAnswer:
    """One answer to one of the three onboarding prompts.

    ``typed=True`` marks the text fallback: the user answered in writing instead of voice, which
    the persona extractor should weight differently (voice carries pacing and humor timing).
    """

    prompt_index: int
    text: str
    audio_url: str | None = None
    typed: bool = False


@dataclass(frozen=True)
class OnboardingEvent:
    kind: Literal["interview_complete"]
    user_id: str
    answers: tuple[InterviewAnswer, ...]


@dataclass(frozen=True)
class DirectedRating:
    """One directed row: rater's view of ratee. a→b and b→a are separate rows (§1 rule 2)."""

    rater: str
    ratee: str
    would_see_again: bool
    felt_connected_1_7: int | None
    was_off: bool = False


@dataclass(frozen=True)
class PruneRequest:
    """One attendee's morning-after ratings are in; assembly decides who stays."""

    group_id: str
    plan_id: str
    rater: str
    ratings: tuple[DirectedRating, ...]


@dataclass(frozen=True)
class BackfillRequest:
    """Seats opened by dropping unpaid members; assembly fills them from the queue."""

    group_id: str
    plan_id: str
    seats: int
    dropped: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LegibilityRequest:
    """User asked what the bot knows about them (§10); persona renders the answer."""

    user_id: str


BotEvent = OnboardingEvent | PruneRequest | BackfillRequest | LegibilityRequest
