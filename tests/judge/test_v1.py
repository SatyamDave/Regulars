import math

import pytest

from regulars.core.features import (
    FEATURE_NAMES,
    FOLLOW_UP_RATE,
    LATENCY_NORM_MEDIAN,
    LSM_MEAN,
    REPLY_WEIGHT,
    PairFeatures,
)
from regulars.judge.v1 import DEFAULT_WEIGHTS, JudgeV1, ZScaler, pair_score


def _pf(a: str, b: str, **features: float) -> PairFeatures:
    return PairFeatures(user_a=a, user_b=b, features=features)


POPULATION = [
    _pf("a", "b", follow_up_rate=0.2, lsm_mean=0.7, latency_norm_median=-0.5),
    _pf("b", "a", follow_up_rate=0.4, lsm_mean=0.8, latency_norm_median=0.5),
    _pf("c", "d", follow_up_rate=0.6, lsm_mean=0.9),  # sim transcript: no latency
    _pf("d", "c", follow_up_rate=0.0, lsm_mean=0.6),
]


def test_zscaler_fits_observed_features_and_skips_missing() -> None:
    scaler = ZScaler.fit(POPULATION)
    assert scaler.mean[FOLLOW_UP_RATE] == pytest.approx(0.3)
    assert scaler.n[FOLLOW_UP_RATE] == 4
    # latency is real-only: fit on the two pairs that have it
    assert scaler.n[LATENCY_NORM_MEDIAN] == 2
    assert scaler.mean[LATENCY_NORM_MEDIAN] == pytest.approx(0.0)
    assert scaler.std[LATENCY_NORM_MEDIAN] == pytest.approx(0.5)
    # never observed: neutral parameters, n = 0
    assert scaler.n[REPLY_WEIGHT] == 0
    assert scaler.mean[REPLY_WEIGHT] == 0.0 and scaler.std[REPLY_WEIGHT] == 1.0
    assert set(scaler.mean) == set(FEATURE_NAMES)


def test_zscaler_std_floor_for_constant_feature() -> None:
    scaler = ZScaler.fit([_pf("a", "b", lsm_mean=0.5), _pf("b", "a", lsm_mean=0.5)])
    assert scaler.std[LSM_MEAN] == 1e-6
    z, _ = scaler.transform(_pf("x", "y", lsm_mean=0.5))
    assert z[LSM_MEAN] == 0.0


def test_missing_feature_imputed_to_mean_and_recorded() -> None:
    scaler = ZScaler.fit(POPULATION)
    z, missing = scaler.transform(POPULATION[2])
    assert LATENCY_NORM_MEDIAN in missing
    assert z[LATENCY_NORM_MEDIAN] == 0.0
    assert FOLLOW_UP_RATE not in missing
    assert z[FOLLOW_UP_RATE] == pytest.approx((0.6 - 0.3) / scaler.std[FOLLOW_UP_RATE])


def test_judge_output_missing_lists_only_weighted_features() -> None:
    judge = JudgeV1.default(POPULATION)
    out = judge.score(POPULATION[2])
    assert LATENCY_NORM_MEDIAN in out.missing
    assert all(name in judge.weights for name in out.missing)
    assert out.contributions[LATENCY_NORM_MEDIAN] == 0.0


def test_contributions_sum_to_y_hat() -> None:
    judge = JudgeV1.default(POPULATION)
    for pf in POPULATION:
        out = judge.score(pf)
        assert math.fsum(out.contributions.values()) == pytest.approx(out.y_hat, abs=1e-12)
        assert set(out.contributions) == set(DEFAULT_WEIGHTS)


def test_default_weights_are_known_features_with_documented_signs() -> None:
    assert set(DEFAULT_WEIGHTS) <= set(FEATURE_NAMES)
    assert DEFAULT_WEIGHTS[FOLLOW_UP_RATE] > 0
    assert DEFAULT_WEIGHTS[LATENCY_NORM_MEDIAN] < 0  # faster than own baseline is better
    assert DEFAULT_WEIGHTS["disclosure_asymmetry"] < 0
    assert DEFAULT_WEIGHTS["question_gini"] < 0
    assert 0 < DEFAULT_WEIGHTS[LSM_MEAN] < DEFAULT_WEIGHTS["lsm_slope"]


def test_unknown_weight_key_rejected() -> None:
    with pytest.raises(ValueError, match="unknown features"):
        JudgeV1(weights={"charisma": 1.0}, scaler=ZScaler.fit(POPULATION))


def test_faster_latency_scores_higher() -> None:
    judge = JudgeV1.default(POPULATION)
    fast = judge.score(_pf("a", "b", follow_up_rate=0.3, lsm_mean=0.75, latency_norm_median=-1))
    slow = judge.score(_pf("a", "b", follow_up_rate=0.3, lsm_mean=0.75, latency_norm_median=1))
    assert fast.y_hat > slow.y_hat


def test_pair_score_is_mean_of_directions() -> None:
    judge = JudgeV1.default(POPULATION)
    ab, ba = judge.score(POPULATION[0]), judge.score(POPULATION[1])
    assert pair_score(ab, ba) == pytest.approx((ab.y_hat + ba.y_hat) / 2)


def test_json_round_trip_gives_identical_scores_and_version() -> None:
    judge = JudgeV1.default(POPULATION, version="v1.0-test")
    clone = JudgeV1.from_json(judge.to_json())
    assert clone == judge
    assert clone.judge_version == judge.judge_version
    assert clone.judge_version.startswith("v1.0-test@")
    for pf in POPULATION:
        assert clone.score(pf) == judge.score(pf)


def test_fingerprint_changes_with_weights() -> None:
    judge = JudgeV1.default(POPULATION)
    other = judge.model_copy(update={"weights": {**judge.weights, FOLLOW_UP_RATE: 2.0}})
    assert other.fingerprint() != judge.fingerprint()
