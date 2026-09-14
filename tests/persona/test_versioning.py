from datetime import UTC, datetime

import pytest

from regulars.core import PersonaSpec
from regulars.persona.versioning import (
    OBSERVED_CONFIDENCE,
    PersonaVersion,
    merge_update,
    next_version,
)

T0 = datetime(2026, 9, 1, tzinfo=UTC)


def _spec(**social: object) -> PersonaSpec:
    return PersonaSpec.model_validate(
        {
            "social": {"friction_points": ["being talked over"], **social},
            "stated": {"exclusions": ["no crypto people"], "seeking": "a standing table"},
            "provenance": {
                "sources": ["voice_interview"],
                "confidence_by_field": {"social.friction_points": 0.6},
            },
        }
    )


def test_first_version_needs_user_id() -> None:
    with pytest.raises(ValueError, match="user_id is required"):
        next_version([], _spec(), "voice_interview")
    v1 = next_version([], _spec(), "voice_interview", user_id="u1", now=T0)
    assert v1 == PersonaVersion(
        user_id="u1", version=1, spec=_spec(), source="voice_interview", created_at=T0
    )


def test_increments_and_never_mutates() -> None:
    spec = _spec()
    v1 = next_version([], spec, "voice_interview", user_id="u1", now=T0)
    v3 = PersonaVersion(user_id="u1", version=3, spec=spec, source="chat_behavior", created_at=T0)
    existing = [v3, v1]  # out of order on purpose
    snapshot = [v.model_copy(deep=True) for v in existing]

    v4 = next_version(existing, spec, "user_correction")
    assert v4.version == 4
    assert v4.user_id == "u1"
    assert v4.created_at.tzinfo is not None
    assert existing == snapshot
    assert v4.spec == spec and v4.spec is not spec
    v4.spec.social.friction_points.append("mutated copy")
    assert spec.social.friction_points == ["being talked over"]


def test_user_id_conflicts_rejected() -> None:
    v1 = next_version([], _spec(), "voice_interview", user_id="u1", now=T0)
    with pytest.raises(ValueError, match="does not match"):
        next_version([v1], _spec(), "chat_behavior", user_id="u2")
    other = v1.model_copy(update={"user_id": "u2"})
    with pytest.raises(ValueError, match="several users"):
        next_version([v1, other], _spec(), "chat_behavior")


def test_version_is_frozen_and_aware() -> None:
    v1 = next_version([], _spec(), "voice_interview", user_id="u1", now=T0)
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises its own frozen error
        v1.version = 9  # type: ignore[misc]
    with pytest.raises(ValueError, match="timezone-aware"):
        PersonaVersion(
            user_id="u",
            version=1,
            spec=_spec(),
            source="chatlog_import",
            created_at=datetime(2026, 1, 1),
        )


def test_merge_update_never_drops_exclusions() -> None:
    base = _spec()
    observed = PersonaSpec.model_validate(
        {
            "conversational": {"avg_turn_length": "short", "opener_style": ""},
            "social": {"friction_points": ["Being Talked Over", "one-upping"], "drains": ["noise"]},
            "stated": {"exclusions": [], "seeking": ""},
            "provenance": {"confidence_by_field": {"social.drains": 0.8}},
        }
    )
    base_snapshot = base.model_copy(deep=True)
    merged = merge_update(base, observed, "chat_behavior")

    assert base == base_snapshot
    assert merged.stated.exclusions == ["no crypto people"]
    assert merged.stated.seeking == "a standing table"
    assert merged.social.friction_points == ["being talked over", "one-upping"]  # deduped
    assert merged.social.drains == ["noise"]
    assert merged.conversational.avg_turn_length == "short"  # explicitly observed
    assert merged.conversational.opener_style == ""  # empty observed value ignored
    assert merged.provenance.sources == ["voice_interview", "chat_behavior"]
    conf = merged.provenance.confidence_by_field
    assert conf["social.friction_points"] == 0.6  # base rating kept for a grown list
    assert conf["social.drains"] == 0.8
    assert conf["conversational.avg_turn_length"] == OBSERVED_CONFIDENCE


def test_merge_update_unset_scalars_keep_base() -> None:
    base = PersonaSpec.model_validate(
        {"conversational": {"avg_turn_length": "long", "humor_register": "dry"}}
    )
    observed = PersonaSpec()  # nothing explicitly set
    merged = merge_update(base, observed, "chat_behavior")
    assert merged.conversational.avg_turn_length == "long"
    assert merged.conversational.humor_register == "dry"
    assert merged.provenance.sources == ["chat_behavior"]
