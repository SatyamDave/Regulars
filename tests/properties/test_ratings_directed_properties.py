"""Property-based tests for directed ratings (§1 rule 2, regulars/db/repo.py insert_rating).

For any generated set of (rater, ratee, value) triples with rater != ratee inserted via
``insert_rating`` into a fresh SQLite engine: a->b and b->a are independently retrievable and
never collapsed/averaged. Inserting rater==ratee always raises, for ALL such inputs.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import Engine

from regulars.db import repo
from regulars.db.engine import sqlite_memory_engine

USER_IDS = ["u1", "u2", "u3", "u4"]


def _fresh_setup() -> tuple[Engine, repo.Plan]:
    engine = sqlite_memory_engine()
    with engine.begin() as conn:
        for uid in USER_IDS:
            repo.create_user(conn, phone_hash=f"h-{uid}", zip_code="94110", user_id=uid)
        group = repo.create_group(conn, zip_code="94110", assembly_method="model")
        plan = repo.create_plan(conn, group.id)
    return engine, plan


@st.composite
def directed_triples(draw: st.DrawFn) -> list[tuple[str, str, bool]]:
    pairs = draw(
        st.lists(
            st.tuples(
                st.sampled_from(USER_IDS),
                st.sampled_from(USER_IDS),
                st.booleans(),
            ),
            min_size=1,
            max_size=6,
            unique_by=lambda t: (t[0], t[1]),
        )
    )
    return [(a, b, v) for (a, b, v) in pairs if a != b]


@given(triples=directed_triples())
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_directed_ratings_never_collapsed(triples: list[tuple[str, str, bool]]) -> None:
    engine, plan = _fresh_setup()
    try:
        with engine.begin() as conn:
            for rater, ratee, value in triples:
                repo.insert_rating(
                    conn, plan.id, rater_id=rater, ratee_id=ratee, would_see_again=value
                )
            stored = {
                (r.rater_id, r.ratee_id): r.would_see_again
                for r in repo.ratings_for_plan(conn, plan.id)
            }
        for rater, ratee, value in triples:
            # a->b is independently retrievable with its own value
            assert stored[(rater, ratee)] == value
            # the reverse direction, if present, is a distinct row (never averaged into this one)
            reverse = stored.get((ratee, rater))
            if reverse is not None:
                # both directions coexist as separate keys; nothing here collapses them.
                assert (ratee, rater) in stored
    finally:
        engine.dispose()


@given(
    rater=st.sampled_from(USER_IDS),
    ratee=st.sampled_from(USER_IDS),
    value=st.booleans(),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_self_rating_always_rejected(rater: str, ratee: str, value: bool) -> None:
    engine, plan = _fresh_setup()
    try:
        with engine.begin() as conn:
            if rater == ratee:
                with pytest.raises(ValueError, match="directed"):
                    repo.insert_rating(
                        conn, plan.id, rater_id=rater, ratee_id=ratee, would_see_again=value
                    )
            else:
                # sanity: the non-self case must succeed, so the property discriminates
                rating = repo.insert_rating(
                    conn, plan.id, rater_id=rater, ratee_id=ratee, would_see_again=value
                )
                assert rating.rater_id == rater
                assert rating.ratee_id == ratee
    finally:
        engine.dispose()


@given(
    a=st.sampled_from(USER_IDS),
    b=st.sampled_from(USER_IDS),
    value_ab=st.booleans(),
    value_ba=st.booleans(),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_both_directions_independent_even_with_opposite_values(
    a: str, b: str, value_ab: bool, value_ba: bool
) -> None:
    if a == b:
        return
    engine, plan = _fresh_setup()
    try:
        with engine.begin() as conn:
            repo.insert_rating(conn, plan.id, rater_id=a, ratee_id=b, would_see_again=value_ab)
            repo.insert_rating(conn, plan.id, rater_id=b, ratee_id=a, would_see_again=value_ba)
            rows = {
                (r.rater_id, r.ratee_id): r.would_see_again
                for r in repo.ratings_for_plan(conn, plan.id)
            }
        assert rows[(a, b)] == value_ab
        assert rows[(b, a)] == value_ba
        # even when the two directed values disagree, both rows persist untouched -- no averaging
        if value_ab != value_ba:
            assert rows[(a, b)] != rows[(b, a)]
    finally:
        engine.dispose()
