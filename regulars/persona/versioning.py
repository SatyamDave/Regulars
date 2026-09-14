"""Append-only persona versions, as pure values (§1 rule 1).

Nothing here touches the database: :func:`next_version` computes the record that
``regulars.db.repo.append_persona_version`` would store, and :func:`merge_update` folds observed
chat behavior into a base spec without ever dropping something the user stated.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from regulars.core.persona import PersonaSource, PersonaSpec, Provenance
from regulars.persona._fields import SECTIONS, get_value

OBSERVED_CONFIDENCE = 0.5
"""Confidence for a field taken from observation when the observed spec did not rate it."""


class PersonaVersion(BaseModel, frozen=True):
    user_id: str
    version: int = Field(ge=1)
    spec: PersonaSpec
    source: PersonaSource
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return v


def next_version(
    existing: Sequence[PersonaVersion],
    spec: PersonaSpec,
    source: PersonaSource,
    *,
    user_id: str | None = None,
    now: datetime | None = None,
) -> PersonaVersion:
    """Build version ``max(existing)+1`` (or 1). Inputs are not mutated; ``spec`` is deep-copied.

    ``user_id`` is taken from ``existing`` when present and must agree with the argument if both
    are given; it is required when ``existing`` is empty.
    """
    ids = {v.user_id for v in existing}
    if len(ids) > 1:
        raise ValueError(f"existing versions span several users: {sorted(ids)}")
    if ids:
        (owner,) = ids
        if user_id is not None and user_id != owner:
            raise ValueError(f"user_id {user_id!r} does not match existing versions ({owner!r})")
    elif user_id is None:
        raise ValueError("user_id is required when there are no existing versions")
    else:
        owner = user_id
    number = max((v.version for v in existing), default=0) + 1
    return PersonaVersion(
        user_id=owner,
        version=number,
        spec=spec.model_copy(deep=True),
        source=source,
        created_at=now or datetime.now(UTC),
    )


def _union(base: list[str], extra: list[str]) -> list[str]:
    seen = {x.strip().casefold() for x in base}
    out = list(base)
    for x in extra:
        key = x.strip().casefold()
        if key and key not in seen:
            seen.add(key)
            out.append(x)
    return out


def merge_update(base: PersonaSpec, observed: PersonaSpec, source: PersonaSource) -> PersonaSpec:
    """Fold ``observed`` (e.g. from chat behavior) into ``base``; returns a new spec.

    Rules:

    * List fields are unioned (base order first, case-insensitive dedupe). Nothing is removed.
    * Scalar fields take the observed value only when ``observed`` set that field explicitly
      (pydantic ``model_fields_set``) and it is non-empty; otherwise the base value stays.
    * ``stated`` is the user's own words: ``exclusions`` can only grow and ``seeking`` changes
      only when observed explicitly sets a non-empty value. Behavioral observation never
      overrides an exclusion.
    * ``provenance.sources`` gains ``source``; confidence is copied for fields taken from
      ``observed`` (falling back to :data:`OBSERVED_CONFIDENCE`), kept from base otherwise.
    """
    data: dict[str, Any] = base.model_dump()
    confidence = dict(base.provenance.confidence_by_field)
    for section in SECTIONS:
        obs_section = getattr(observed, section)
        explicit = obs_section.model_fields_set
        for field in type(obs_section).model_fields:
            path = f"{section}.{field}"
            obs_val = get_value(observed, path)
            base_val = get_value(base, path)
            if isinstance(base_val, list):
                merged = _union(base_val, obs_val)
                if merged != base_val:
                    data[section][field] = merged
                    confidence[path] = observed.provenance.confidence_by_field.get(
                        path, confidence.get(path, OBSERVED_CONFIDENCE)
                    )
                continue
            if field in explicit and obs_val not in ("", None) and obs_val != base_val:
                data[section][field] = obs_val
                confidence[path] = observed.provenance.confidence_by_field.get(
                    path, OBSERVED_CONFIDENCE
                )
    sources = list(base.provenance.sources)
    if source not in sources:
        sources.append(source)
    data["provenance"] = Provenance(sources=sources, confidence_by_field=confidence).model_dump()
    return PersonaSpec.model_validate(data)
