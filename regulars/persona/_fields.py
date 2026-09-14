"""Field-path helpers shared by extraction, versioning, legibility and corrections.

A *field path* is ``"<section>.<field>"``, e.g. ``"social.friction_points"``. The four content
sections of :class:`~regulars.core.persona.PersonaSpec` are enumerated from the pydantic model
itself so this module cannot drift from ``regulars.core``. ``provenance`` is never a field path:
it is metadata about the other fields, not a fact about the user.
"""

from __future__ import annotations

from typing import Any

from regulars.core.persona import PersonaSpec

SECTIONS: tuple[str, ...] = ("conversational", "content", "social", "stated")

_DEFAULT_SPEC = PersonaSpec()


def field_paths() -> list[str]:
    """Every ``section.field`` path in spec order."""
    paths: list[str] = []
    for section in SECTIONS:
        model = getattr(_DEFAULT_SPEC, section)
        paths.extend(f"{section}.{name}" for name in type(model).model_fields)
    return paths


def split_path(path: str) -> tuple[str, str]:
    """Validate and split a field path. Raises ``KeyError`` for anything not in the spec."""
    section, _, field = path.partition(".")
    if section not in SECTIONS or not field:
        raise KeyError(f"unknown persona field path {path!r}")
    model = getattr(_DEFAULT_SPEC, section)
    if field not in type(model).model_fields:
        raise KeyError(f"unknown persona field path {path!r}")
    return section, field


def get_value(spec: PersonaSpec, path: str) -> Any:
    section, field = split_path(path)
    return getattr(getattr(spec, section), field)


def default_value(path: str) -> Any:
    return get_value(_DEFAULT_SPEC, path)


def is_populated(spec: PersonaSpec, path: str) -> bool:
    """True when the field differs from a blank spec or extraction recorded confidence for it.

    The second clause matters for ``Literal`` fields whose extracted value happens to equal the
    default (e.g. ``avg_turn_length == "medium"``): the LLM asserted it, so it is known.
    """
    return (
        get_value(spec, path) != default_value(path) or path in spec.provenance.confidence_by_field
    )
