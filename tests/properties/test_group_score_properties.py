"""Property-based tests for regulars/assembly/optimizer.py: group_score is the min, never mean.

For any generated ``PairScores`` and any generated group of >= 2 members, ``group_score`` equals
the minimum pair score over the group, never the mean. Includes an explicit counter-example class
of inputs (one very low outlier pair) where mean and min diverge, to make sure the property
actually discriminates between the two.
"""

from __future__ import annotations

import math
from itertools import combinations

from hypothesis import given, settings
from hypothesis import strategies as st

from regulars.assembly.optimizer import PairScores, group_score

_finite_score = st.floats(
    min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False, width=32
)


@st.composite
def scored_group(
    draw: st.DrawFn, min_size: int = 2, max_size: int = 6
) -> tuple[list[str], PairScores]:
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    members = [f"u{i}" for i in range(n)]
    scores: dict[tuple[str, str], float] = {}
    for a, b in combinations(members, 2):
        scores[(a, b)] = draw(_finite_score)
    return members, PairScores(scores)


@given(data=scored_group())
@settings(max_examples=200)
def test_group_score_equals_min_never_mean(data: tuple[list[str], PairScores]) -> None:
    members, scores = data
    pair_values = [scores.get(a, b) for a, b in combinations(members, 2)]
    expected_min = min(pair_values)
    expected_mean = sum(pair_values) / len(pair_values)

    assert group_score(members, scores) == expected_min

    # discriminate: whenever min != mean, group_score must NOT equal the mean either.
    if not math.isclose(expected_min, expected_mean):
        assert group_score(members, scores) != expected_mean


def test_explicit_counter_example_min_and_mean_diverge() -> None:
    """One catastrophic pair should tank the group score even though the mean stays high."""
    members = ["a", "b", "c", "d"]
    scores = PairScores(
        {
            ("a", "b"): 0.95,
            ("a", "c"): 0.90,
            ("a", "d"): 0.92,
            ("b", "c"): 0.93,
            ("b", "d"): 0.91,
            ("c", "d"): -5.0,  # one bad dyad
        }
    )
    pair_values = [scores.get(x, y) for x, y in combinations(members, 2)]
    mean = sum(pair_values) / len(pair_values)
    assert group_score(members, scores) == -5.0
    assert mean > -5.0  # the mean is pulled well above the min by the five good dyads
    assert group_score(members, scores) != mean


@given(
    n=st.integers(min_value=2, max_value=6),
    low_value=st.floats(min_value=-10.0, max_value=-1.0, allow_nan=False, allow_infinity=False),
    high_value=st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_single_low_outlier_always_determines_group_score(
    n: int, low_value: float, high_value: float
) -> None:
    """Counter-example class: all pairs high except one very low pair; min must track the
    outlier while the mean would stay dominated by the high values."""
    members = [f"u{i}" for i in range(n)]
    scores_dict: dict[tuple[str, str], float] = {}
    pairs = list(combinations(members, 2))
    for a, b in pairs:
        scores_dict[(a, b)] = high_value
    outlier_pair = pairs[0]
    scores_dict[outlier_pair] = low_value
    scores = PairScores(scores_dict)

    result = group_score(members, scores)
    assert result == low_value

    pair_values = list(scores_dict.values())
    mean = sum(pair_values) / len(pair_values)
    if len(pairs) > 1:
        assert mean > low_value  # mean is pulled up by the other high-value pairs
        assert result != mean
