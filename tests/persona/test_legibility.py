import re

import pytest
from pydantic import ValidationError

from regulars.core import PersonaSpec
from regulars.persona._fields import field_paths
from regulars.persona.legibility import (
    EMPTY,
    apply_correction,
    describe_field,
    numbered_fields,
    render_plain_english,
)

FULL = PersonaSpec.model_validate(
    {
        "conversational": {
            "opener_style": "A specific question about your week",
            "avg_turn_length": "short",
            "humor_register": "dry",
            "question_tendency": "high",
            "interrupt_tendency": "waits, then lands one line",
            "silence_tolerance": "fine with a lull",
        },
        "content": {
            "unprompted_topics": ["cooking", "hiking"],
            "avoided_topics": ["work"],
            "escalates_on": ["hosting"],
            "deflects_on": ["dating"],
        },
        "social": {
            "energy": "initiator",
            "disclosure_speed": "fast",
            "friction_points": ["being talked over", "one-upping", "phones at the table"],
            "drains": ["small talk"],
        },
        "stated": {"exclusions": ["no crypto people"], "seeking": "A standing table"},
    }
)


def test_renders_every_populated_field_numbered() -> None:
    out = render_plain_english(FULL)
    lines = [ln for ln in out.splitlines() if re.match(r"^\d+\. ", ln)]
    assert len(lines) == len(field_paths()) == 16
    assert [int(ln.split(".")[0]) for ln in lines] == list(range(1, 17))
    for path in field_paths():
        section, field = path.split(".")
        value = getattr(getattr(FULL, section), field)
        assert describe_field(path, value) in out
    assert "being talked over, one-upping and phones at the table" in out
    assert "You told me you don't want: no crypto people." in out
    assert "3 is wrong" in out  # tells the user how to correct
    assert not re.search(r"\b(spec|persona|field|provenance|confidence)\b", out.lower())
    assert "_" not in out


def test_only_populated_fields_render() -> None:
    sparse = PersonaSpec.model_validate({"social": {"drains": ["small talk"]}})
    rows = numbered_fields(sparse)
    assert rows == [(1, "social.drains", "What wears you out: small talk.")]
    assert render_plain_english(PersonaSpec()) == EMPTY


def test_asserted_default_literal_counts_as_populated() -> None:
    spec = PersonaSpec.model_validate(
        {"provenance": {"confidence_by_field": {"conversational.avg_turn_length": 0.6}}}
    )
    assert [p for _, p, _ in numbered_fields(spec)] == ["conversational.avg_turn_length"]


def test_apply_correction_is_non_destructive() -> None:
    before = FULL.model_copy(deep=True)
    fixed = apply_correction(FULL, "conversational.question_tendency", "low")
    assert before == FULL
    assert fixed.conversational.question_tendency == "low"
    assert fixed.provenance.sources == ["user_correction"]
    assert fixed.provenance.confidence_by_field["conversational.question_tendency"] == 1.0
    # everything else untouched
    assert fixed.model_dump(exclude={"conversational", "provenance"}) == FULL.model_dump(
        exclude={"conversational", "provenance"}
    )

    twice = apply_correction(fixed, "social.friction_points", ["being ignored"])
    assert twice.provenance.sources == ["user_correction"]  # not duplicated
    assert twice.social.friction_points == ["being ignored"]
    assert fixed.social.friction_points == FULL.social.friction_points


def test_apply_correction_validates() -> None:
    with pytest.raises(KeyError):
        apply_correction(FULL, "provenance.sources", ["x"])
    with pytest.raises(KeyError):
        apply_correction(FULL, "social.vibes", "good")
    with pytest.raises(ValidationError):
        apply_correction(FULL, "social.energy", "loud")


def test_correction_by_line_number_round_trip() -> None:
    rows = numbered_fields(FULL)
    number, path, _ = rows[2]  # "3 is wrong"
    assert number == 3
    fixed = apply_correction(FULL, path, "absurdist" if path.endswith("humor_register") else "x")
    assert fixed.provenance.confidence_by_field[path] == 1.0
