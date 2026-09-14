"""Sim scenarios (§3). Three scenes of distinct social difficulty; one easy scene alone yields
uniformly pleasant transcripts and no variance.

``mild_friction`` names a concrete disagreement drawn from the pair's ``friction_points`` /
``escalates_on``; call :meth:`Scenario.for_pair` to render it. ``dead_air`` carries the dying
exchange as seed turns so the transcript contains the flat moment the §5 repair features look for.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from regulars.core.persona import PersonaSpec

Difficulty = Literal["baseline", "repair", "friction"]

DISAGREEMENT_PLACEHOLDER = "{disagreement}"

# Used when neither persona spec carries friction_points or escalates_on. Concrete on purpose:
# an abstract "you disagree about something" produces nothing.
FALLBACK_DISAGREEMENT = "whether it's rude to keep your phone face-up on the table during dinner"


@dataclass(frozen=True)
class Scenario:
    id: str
    setup: str
    difficulty: Difficulty
    injects_friction: bool
    # Lines spoken before the sim starts, alternating from the opener. Empty for most scenes.
    seed_turns: tuple[str, ...] = ()

    def for_pair(self, spec_a: PersonaSpec, spec_b: PersonaSpec) -> Scenario:
        """Render pair-specific placeholders. Idempotent for scenes without placeholders."""
        if DISAGREEMENT_PLACEHOLDER not in self.setup:
            return self
        topic = disagreement_for(spec_a, spec_b)
        return replace(self, setup=self.setup.replace(DISAGREEMENT_PLACEHOLDER, topic))


def disagreement_for(spec_a: PersonaSpec, spec_b: PersonaSpec) -> str:
    """The first concrete thing either persona pushes back on. Persona A's list is checked first,
    then B's, friction_points before escalates_on, so the same pair always gets the same topic."""
    for spec in (spec_a, spec_b):
        for item in (*spec.social.friction_points, *spec.content.escalates_on):
            text = item.strip()
            if text:
                return text
    return FALLBACK_DISAGREEMENT


COLD_OPEN = Scenario(
    id="cold_open",
    setup=(
        "You have just sat down across from {counterpart} at a small table in a neighborhood "
        "restaurant. It is the first ten minutes. Drinks have not been ordered yet. You have "
        "never met. Nobody has told you what to talk about."
    ),
    difficulty="baseline",
    injects_friction=False,
)

DEAD_AIR = Scenario(
    id="dead_air",
    setup=(
        "You are twenty minutes into dinner with {counterpart}. A topic has just died. The last "
        "few lines are below. There is a silence. Nobody owes anybody a rescue: if you would let "
        "it sit, let it sit; if you would change the subject, do it the way you actually would."
    ),
    difficulty="repair",
    injects_friction=False,
    seed_turns=(
        "So you said you moved here about two years ago?",
        "Yeah. Two years in March.",
        "Cool. Cool.",
        "Yeah.",
    ),
)

MILD_FRICTION = Scenario(
    id="mild_friction",
    setup=(
        "You are half an hour into dinner with {counterpart}. Mains have arrived. A disagreement "
        f"has just surfaced about {DISAGREEMENT_PLACEHOLDER}. You do not agree with them. "
        "Respond the way you actually would: accommodate, push, deflect, or go quiet, whichever "
        "is true to you."
    ),
    difficulty="friction",
    injects_friction=True,
)

SCENARIOS: dict[str, Scenario] = {s.id: s for s in (COLD_OPEN, DEAD_AIR, MILD_FRICTION)}
