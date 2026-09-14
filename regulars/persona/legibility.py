"""Legibility as a feature (§10): "what do you know about me" and corrections.

:func:`render_plain_english` prints every populated field as a numbered, warm, jargon-free line
so the user can reply "3 is wrong, I ask tons of questions". :func:`numbered_fields` exposes the
line-number → field-path mapping the bot needs to turn that reply into
:func:`apply_correction`, which returns a new spec with ``user_correction`` in provenance and
confidence 1.0 for the corrected field. Corrections are free, high-quality labels.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from regulars.core.persona import PersonaSpec, Provenance
from regulars.persona._fields import field_paths, is_populated, split_path

HEADER = (
    "Here's what I've picked up about you so far. If a line is off, reply with its number and "
    'what\'s actually true — like "3 is wrong, I ask tons of questions."'
)
EMPTY = "I don't know much about you yet. Let's do the interview and I'll fill this in."


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:] if s else s


_LITERAL_LINES: dict[str, dict[str, str]] = {
    "conversational.avg_turn_length": {
        "short": "You keep your messages short and punchy.",
        "medium": "Your messages run a middling length — not one-liners, not essays.",
        "long": "You write in long, full messages.",
    },
    "conversational.humor_register": {
        "dry": "Your humor is dry.",
        "absurdist": "Your humor runs absurdist.",
        "self-deprecating": "Your humor is self-deprecating.",
        "none": "You don't lean on jokes much.",
    },
    "conversational.question_tendency": {
        "high": "You ask a lot of questions.",
        "low": "You don't ask many questions; you tend to share instead.",
    },
    "social.energy": {
        "initiator": "In a group, you're the one who gets things going.",
        "responder": "In a group, you tend to respond rather than start things.",
        "observer": "In a group, you mostly watch and weigh in when it counts.",
    },
    "social.disclosure_speed": {
        "fast": "You open up quickly.",
        "slow": "You take your time opening up.",
    },
}

_TEXT_LINES: dict[str, Callable[[Any], str]] = {
    "conversational.opener_style": lambda v: f"You tend to open with {_lower_first(v)}",
    "conversational.interrupt_tendency": lambda v: f"On jumping in: {_lower_first(v)}",
    "conversational.silence_tolerance": lambda v: f"On quiet moments: {_lower_first(v)}",
    "content.unprompted_topics": lambda v: f"Things you bring up on your own: {_join(v)}.",
    "content.avoided_topics": lambda v: f"Things you steer away from: {_join(v)}.",
    "content.escalates_on": lambda v: f"Things that get you fired up: {_join(v)}.",
    "content.deflects_on": lambda v: f"Things you'd rather not get into: {_join(v)}.",
    "social.friction_points": lambda v: f"What rubs you the wrong way: {_join(v)}.",
    "social.drains": lambda v: f"What wears you out: {_join(v)}.",
    "stated.exclusions": lambda v: f"You told me you don't want: {_join(v)}.",
    "stated.seeking": lambda v: f"What you're looking for: {_lower_first(v)}",
}


def describe_field(path: str, value: Any) -> str:
    """One plain-English sentence for a field value."""
    if path in _LITERAL_LINES:
        return _LITERAL_LINES[path][str(value)]
    return _TEXT_LINES[path](value)


def numbered_fields(spec: PersonaSpec) -> list[tuple[int, str, str]]:
    """``(line_number, field_path, sentence)`` for every populated field, 1-based, spec order."""
    out: list[tuple[int, str, str]] = []
    for path in field_paths():
        if not is_populated(spec, path):
            continue
        section, field = split_path(path)
        value = getattr(getattr(spec, section), field)
        out.append((len(out) + 1, path, describe_field(path, value)))
    return out


def render_plain_english(spec: PersonaSpec) -> str:
    lines = numbered_fields(spec)
    if not lines:
        return EMPTY
    body = "\n".join(f"{n}. {sentence}" for n, _, sentence in lines)
    return f"{HEADER}\n\n{body}"


def apply_correction(spec: PersonaSpec, field_path: str, new_value: Any) -> PersonaSpec:
    """Return a new spec with ``field_path`` set to ``new_value`` (validated against the spec).

    ``provenance.sources`` gains ``"user_correction"`` and the field's confidence becomes 1.0.
    The input spec is untouched. Raises ``KeyError`` for an unknown path and pydantic's
    ``ValidationError`` for a value the field does not accept.
    """
    section, field = split_path(field_path)
    data = spec.model_dump()
    data[section][field] = new_value
    sources = list(spec.provenance.sources)
    if "user_correction" not in sources:
        sources.append("user_correction")
    confidence = {**spec.provenance.confidence_by_field, field_path: 1.0}
    data["provenance"] = Provenance(sources=sources, confidence_by_field=confidence).model_dump()
    return PersonaSpec.model_validate(data)
