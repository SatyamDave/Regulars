"""Property-based tests for regulars/persona/versioning.py (§1 rule 1).

For any sequence of persona spec updates applied via ``next_version``/``merge_update``:

* versions are strictly increasing integers starting at 1 (append-only versioning contract).
* no version's ``spec`` object is ever mutated after creation (compared against a snapshot taken
  right after creation).
* ``stated.exclusions`` never shrinks across ``merge_update`` calls (the merge_update contract:
  behavioral observation never overrides an exclusion).
"""

from __future__ import annotations

from datetime import UTC, datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from regulars.core.persona import PersonaSource, PersonaSpec
from regulars.persona.versioning import PersonaVersion, merge_update, next_version

PERSONA_SOURCES: tuple[PersonaSource, ...] = (
    "voice_interview",
    "chat_behavior",
    "chatlog_import",
    "user_correction",
)

_word = st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=8)


@st.composite
def persona_specs(draw: st.DrawFn) -> PersonaSpec:
    exclusions = draw(st.lists(_word, max_size=5, unique=True))
    friction_points = draw(st.lists(_word, max_size=5, unique=True))
    seeking = draw(st.one_of(st.just(""), _word))
    return PersonaSpec.model_validate(
        {
            "social": {"friction_points": friction_points},
            "stated": {"exclusions": exclusions, "seeking": seeking},
        }
    )


@st.composite
def source_lists(draw: st.DrawFn, *, min_size: int = 1, max_size: int = 8) -> list[PersonaSource]:
    return draw(st.lists(st.sampled_from(PERSONA_SOURCES), min_size=min_size, max_size=max_size))


# --- versions are strictly increasing integers starting at 1 -----------------------------------


@given(
    specs_and_sources=st.lists(
        st.tuples(persona_specs(), st.sampled_from(PERSONA_SOURCES)), min_size=1, max_size=15
    )
)
@settings(max_examples=200)
def test_versions_strictly_increasing_from_one(
    specs_and_sources: list[tuple[PersonaSpec, PersonaSource]],
) -> None:
    existing: list[PersonaVersion] = []
    for i, (spec, source) in enumerate(specs_and_sources):
        v = next_version(existing, spec, source, user_id="u1", now=datetime.now(UTC))
        assert v.version == i + 1
        existing.append(v)
    versions = [v.version for v in existing]
    assert versions == list(range(1, len(existing) + 1))
    assert versions[0] == 1


@given(source_list=source_lists())
@settings(max_examples=200)
def test_out_of_order_existing_still_yields_next_max_plus_one(
    source_list: list[PersonaSource],
) -> None:
    """``next_version`` uses max(existing)+1 regardless of list order (matches versioning.py)."""
    existing: list[PersonaVersion] = []
    now = datetime.now(UTC)
    for i, source in enumerate(source_list):
        existing.append(
            PersonaVersion(
                user_id="u1", version=i + 1, spec=PersonaSpec(), source=source, created_at=now
            )
        )
    # shuffle order shouldn't matter
    shuffled = list(reversed(existing))
    v_next = next_version(shuffled, PersonaSpec(), "user_correction")
    assert v_next.version == len(existing) + 1


# --- no version's spec object is ever mutated after creation ------------------------------------


@given(
    specs_and_sources=st.lists(
        st.tuples(persona_specs(), st.sampled_from(PERSONA_SOURCES)), min_size=1, max_size=10
    )
)
@settings(max_examples=200)
def test_spec_never_mutated_after_creation(
    specs_and_sources: list[tuple[PersonaSpec, PersonaSource]],
) -> None:
    existing: list[PersonaVersion] = []
    snapshots: list[PersonaSpec] = []
    for spec, source in specs_and_sources:
        v = next_version(existing, spec, source, user_id="u1", now=datetime.now(UTC))
        existing.append(v)
        snapshots.append(v.spec.model_copy(deep=True))
        # the input spec itself must not have been captured by reference either
        assert v.spec is not spec

    # After building the whole chain, every stored version's spec must still equal its snapshot,
    # i.e. nothing downstream (later next_version/merge_update calls) mutated an earlier version.
    for v, snap in zip(existing, snapshots, strict=True):
        assert v.spec == snap


@given(
    base=persona_specs(),
    observed=persona_specs(),
    source=st.sampled_from(PERSONA_SOURCES),
)
@settings(max_examples=200)
def test_merge_update_does_not_mutate_base_or_observed(
    base: PersonaSpec, observed: PersonaSpec, source: PersonaSource
) -> None:
    base_snapshot = base.model_copy(deep=True)
    observed_snapshot = observed.model_copy(deep=True)
    merge_update(base, observed, source)
    assert base == base_snapshot
    assert observed == observed_snapshot


# --- stated.exclusions never shrinks across merge_update calls ---------------------------------


@given(
    base=persona_specs(),
    updates=st.lists(st.tuples(persona_specs(), st.sampled_from(PERSONA_SOURCES)), max_size=10),
)
@settings(max_examples=200)
def test_exclusions_never_shrink_across_merges(
    base: PersonaSpec, updates: list[tuple[PersonaSpec, PersonaSource]]
) -> None:
    current = base
    prev_exclusions = {e.strip().casefold() for e in current.stated.exclusions}
    for observed, source in updates:
        current = merge_update(current, observed, source)
        new_exclusions = {e.strip().casefold() for e in current.stated.exclusions}
        assert prev_exclusions <= new_exclusions
        prev_exclusions = new_exclusions
