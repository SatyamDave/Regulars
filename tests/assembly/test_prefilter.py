from collections import Counter

import pytest

from regulars.assembly.prefilter import (
    Candidate,
    pair_key,
    plausible_pairs,
    rejection_reason,
)


def cand(
    uid: str,
    zip_code: str = "94110",
    slots: tuple[str, ...] = ("fri",),
    exclusions: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> Candidate:
    return Candidate(
        user_id=uid,
        zip=zip_code,
        availability=frozenset(slots),
        exclusions=frozenset(exclusions),
        tags=frozenset(tags),
    )


def test_pair_key_is_sorted_and_rejects_self_pair() -> None:
    assert pair_key("b", "a") == ("a", "b") == pair_key("a", "b")
    with pytest.raises(ValueError):
        pair_key("a", "a")


def test_rejects_zip_mismatch_unless_adjacent() -> None:
    a, b = cand("a", "94110"), cand("b", "94103")
    assert rejection_reason(a, b) == "zip"
    assert rejection_reason(a, b, {"94110": {"94103"}}) is None
    # adjacency is symmetrised: declaring it on either side is enough
    assert rejection_reason(a, b, {"94103": {"94110"}}) is None
    assert plausible_pairs([a, b]) == frozenset()
    assert plausible_pairs([a, b], zip_adjacency={"94110": ["94103"]}) == {("a", "b")}


def test_rejects_no_shared_availability() -> None:
    a, b = cand("a", slots=("fri",)), cand("b", slots=("sat",))
    assert rejection_reason(a, b) == "availability"
    assert plausible_pairs([a, b]) == frozenset()
    assert plausible_pairs([a, cand("c", slots=("sat", "fri"))]) == {("a", "c")}


def test_rejects_exclusion_matching_tags_in_either_direction() -> None:
    smoker = cand("s", tags=("smoker",))
    no_smokers = cand("n", exclusions=("smoker",))
    plain = cand("p")
    assert rejection_reason(smoker, no_smokers) == "exclusion"
    assert rejection_reason(no_smokers, smoker) == "exclusion"
    assert plausible_pairs([smoker, no_smokers, plain]) == {("n", "p"), ("p", "s")}


def test_duplicate_candidate_ids_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        plausible_pairs([cand("a"), cand("a")])


def test_cap_per_user_holds_and_is_deterministic() -> None:
    cands = [cand(f"u{i:02d}") for i in range(30)]  # complete graph: 29 partners each
    pairs = plausible_pairs(cands, max_per_user=5, seed=7)
    degree = Counter(u for p in pairs for u in p)
    assert max(degree.values()) <= 5
    assert len(pairs) > 0
    assert plausible_pairs(cands, max_per_user=5, seed=7) == pairs
    # a different seed can pick a different subset, but never breaks the cap
    other = plausible_pairs(cands, max_per_user=5, seed=8)
    assert max(Counter(u for p in other for u in p).values()) <= 5


def test_cap_larger_than_degree_keeps_everything() -> None:
    cands = [cand(f"u{i}") for i in range(6)]
    assert len(plausible_pairs(cands, max_per_user=200)) == 15


def test_cap_must_be_positive() -> None:
    with pytest.raises(ValueError):
        plausible_pairs([cand("a")], max_per_user=0)
