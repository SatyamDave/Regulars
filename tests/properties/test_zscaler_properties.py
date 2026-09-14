"""Property-based tests for regulars/judge/v1.py: ZScaler + JudgeV1 (contributions sum to y_hat).

For any generated list of ``PairFeatures`` with random floats and random missing keys:

* ``ZScaler.fit`` then ``JudgeV1.score`` produces ``contributions`` that sum to ``y_hat`` (within
  float tolerance) for ALL generated inputs.
* z-scored features have the missing-value floor behavior: std never 0 (STD_FLOOR), so z-scoring
  a constant/missing feature never divides by zero.
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from regulars.core.features import FEATURE_NAMES, PairFeatures
from regulars.judge.v1 import DEFAULT_WEIGHTS, STD_FLOOR, JudgeV1, ZScaler

_feature_value = st.floats(
    min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False, width=32
)


@st.composite
def pair_features(draw: st.DrawFn, name_prefix: str = "p") -> PairFeatures:
    present = draw(
        st.lists(st.sampled_from(FEATURE_NAMES), unique=True, max_size=len(FEATURE_NAMES))
    )
    features = {name: draw(_feature_value) for name in present}
    a = draw(st.text(alphabet="abcdefgh", min_size=1, max_size=4))
    b = draw(st.text(alphabet="ijklmnop", min_size=1, max_size=4))
    return PairFeatures(user_a=f"{name_prefix}-{a}", user_b=f"{name_prefix}-{b}", features=features)


@st.composite
def population_and_target(draw: st.DrawFn) -> tuple[list[PairFeatures], PairFeatures]:
    population = draw(st.lists(pair_features(), min_size=1, max_size=8))
    target = draw(pair_features(name_prefix="t"))
    return population, target


@given(data=population_and_target())
@settings(max_examples=200)
def test_contributions_sum_to_y_hat(data: tuple[list[PairFeatures], PairFeatures]) -> None:
    population, target = data
    judge = JudgeV1.default(population)
    out = judge.score(target)
    assert out.y_hat == math.fsum(out.contributions.values())
    assert math.isclose(sum(out.contributions.values()), out.y_hat, rel_tol=1e-9, abs_tol=1e-9)


@given(population=st.lists(pair_features(), min_size=0, max_size=10))
@settings(max_examples=200)
def test_std_never_zero_after_fit(population: list[PairFeatures]) -> None:
    scaler = ZScaler.fit(population)
    for name in FEATURE_NAMES:
        assert scaler.std[name] >= STD_FLOOR
        assert scaler.std[name] > 0.0


@given(data=population_and_target())
@settings(max_examples=200)
def test_missing_features_are_imputed_to_zero_z_and_finite(
    data: tuple[list[PairFeatures], PairFeatures],
) -> None:
    population, target = data
    scaler = ZScaler.fit(population)
    z, missing = scaler.transform(target)
    for name in missing:
        assert z[name] == 0.0
    # every z value is finite: no division producing inf/nan, even for constant/never-seen features
    for value in z.values():
        assert math.isfinite(value)


@given(
    # Kept modest in magnitude: a constant population's mean should reproduce ``value`` exactly,
    # but for very large magnitudes float summation/division can introduce a tiny rounding error
    # that STD_FLOOR (1e-6) then amplifies into a non-zero z. That is a float-precision artifact
    # of the floor, not the property under test, so this range keeps that error negligible.
    value=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    repeats=st.integers(min_value=1, max_value=6),
)
@settings(max_examples=200)
def test_constant_feature_population_hits_std_floor_and_zeroes_out(
    value: float, repeats: int
) -> None:
    population = [
        PairFeatures(user_a="a", user_b="b", features={"lsm_mean": value}) for _ in range(repeats)
    ]
    scaler = ZScaler.fit(population)
    assert scaler.std["lsm_mean"] == STD_FLOOR
    z, missing = scaler.transform(
        PairFeatures(user_a="x", user_b="y", features={"lsm_mean": value})
    )
    assert "lsm_mean" not in missing
    assert math.isclose(z["lsm_mean"], 0.0, abs_tol=1e-6)


def test_default_weights_reference_only_known_features() -> None:
    assert set(DEFAULT_WEIGHTS) <= set(FEATURE_NAMES)
