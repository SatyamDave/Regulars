"""Property-based tests for regulars/assembly/prefilter.py.

For any generated set of candidates, ``plausible_pairs`` never returns a pair that fails ANY hard
constraint (same/adjacent zip, shared availability, no exclusion/tag match), and never exceeds
``max_per_user`` for any single user, across all generated inputs.
"""

from __future__ import annotations

from collections import Counter

from hypothesis import given, settings
from hypothesis import strategies as st

from regulars.assembly.prefilter import Candidate, is_plausible, plausible_pairs

ZIPS = ["94110", "94103", "94107", "10001"]
SLOTS = ["fri", "sat", "sun"]
TAGS = ["vegan", "loud", "quiet", "crypto", "no_kids"]


@st.composite
def candidates(draw: st.DrawFn, min_size: int = 0, max_size: int = 12) -> list[Candidate]:
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    out = []
    for i in range(n):
        zip_code = draw(st.sampled_from(ZIPS))
        availability = frozenset(draw(st.lists(st.sampled_from(SLOTS), max_size=3, unique=True)))
        exclusions = frozenset(draw(st.lists(st.sampled_from(TAGS), max_size=3, unique=True)))
        tags = frozenset(draw(st.lists(st.sampled_from(TAGS), max_size=3, unique=True)))
        out.append(
            Candidate(
                user_id=f"u{i}",
                zip=zip_code,
                availability=availability,
                exclusions=exclusions,
                tags=tags,
            )
        )
    return out


@st.composite
def zip_adjacencies(draw: st.DrawFn) -> dict[str, frozenset[str]]:
    adjacency: dict[str, frozenset[str]] = {}
    for z in ZIPS:
        others = [o for o in ZIPS if o != z]
        neighbors = draw(st.lists(st.sampled_from(others), max_size=2, unique=True))
        if neighbors:
            adjacency[z] = frozenset(neighbors)
    return adjacency


@given(
    cands=candidates(),
    max_per_user=st.integers(min_value=1, max_value=6),
    adjacency=st.one_of(st.none(), zip_adjacencies()),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=200)
def test_plausible_pairs_never_violate_hard_constraints(
    cands: list[Candidate],
    max_per_user: int,
    adjacency: dict[str, frozenset[str]] | None,
    seed: int,
) -> None:
    by_id = {c.user_id: c for c in cands}
    result = plausible_pairs(cands, max_per_user=max_per_user, zip_adjacency=adjacency, seed=seed)
    for a_id, b_id in result:
        a, b = by_id[a_id], by_id[b_id]
        assert is_plausible(a, b, adjacency), f"pair {a_id},{b_id} failed a hard constraint"


@given(
    cands=candidates(),
    max_per_user=st.integers(min_value=1, max_value=6),
    adjacency=st.one_of(st.none(), zip_adjacencies()),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=200)
def test_plausible_pairs_respects_max_per_user_cap(
    cands: list[Candidate],
    max_per_user: int,
    adjacency: dict[str, frozenset[str]] | None,
    seed: int,
) -> None:
    result = plausible_pairs(cands, max_per_user=max_per_user, zip_adjacency=adjacency, seed=seed)
    degree: Counter[str] = Counter()
    for a_id, b_id in result:
        degree[a_id] += 1
        degree[b_id] += 1
    for user_id, count in degree.items():
        assert count <= max_per_user, f"user {user_id} exceeded max_per_user={max_per_user}"


@given(cands=candidates(), seed=st.integers(min_value=0, max_value=1000))
@settings(max_examples=200)
def test_plausible_pairs_are_unordered_and_deduped(cands: list[Candidate], seed: int) -> None:
    """Every returned pair is stored as (a, b) with a < b, and no pair appears twice."""
    result = plausible_pairs(cands, seed=seed)
    seen = set()
    for pair in result:
        a, b = pair
        assert a < b
        assert pair not in seen
        seen.add(pair)
