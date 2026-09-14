import numpy as np
import pytest

from regulars.judge.calibration import MIN_CALIBRATION_N, IsotonicCalibrator


def _data(n: int, seed: int = 0) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(seed)
    y_hat = rng.normal(size=n)
    p = 1 / (1 + np.exp(-1.5 * y_hat))
    y = (rng.random(n) < p).astype(float)
    return y_hat.tolist(), y.tolist()


def test_refuses_fewer_than_floor_pointing_at_spec() -> None:
    y_hat, y = _data(MIN_CALIBRATION_N - 1)
    with pytest.raises(ValueError, match=r"§5"):
        IsotonicCalibrator().fit(y_hat, y)


def test_accepts_exactly_floor() -> None:
    y_hat, y = _data(MIN_CALIBRATION_N)
    IsotonicCalibrator().fit(y_hat, y)


def test_predictions_are_monotone_and_probabilities() -> None:
    y_hat, y = _data(200)
    cal = IsotonicCalibrator().fit(y_hat, y)
    grid = np.linspace(-4, 4, 201)
    p = cal.predict_proba(grid.tolist())
    assert np.all(np.diff(p) >= -1e-12)
    assert p.min() >= 0.0 and p.max() <= 1.0
    # out_of_bounds='clip': far outside the fit range stays at the edge values
    assert cal.predict_proba([-100.0])[0] == pytest.approx(p[0])
    assert cal.predict_proba([100.0])[0] == pytest.approx(p[-1])


def test_brier_beats_coin_flip_on_informative_scores() -> None:
    y_hat, y = _data(500)
    cal = IsotonicCalibrator().fit(y_hat, y)
    assert cal.brier() < 0.25
    assert cal.brier(y_hat, y) == pytest.approx(cal.brier())


def test_reliability_table_bins_cover_data_and_are_calibrated_in_sample() -> None:
    y_hat, y = _data(1000)
    cal = IsotonicCalibrator().fit(y_hat, y)
    rows = cal.reliability_table(bins=10)
    assert sum(r.n for r in rows) == 1000
    for r in rows:
        assert r.lower <= r.mean_predicted <= r.upper + 1e-12
        # isotonic fit is in-sample calibrated per step; bins pool steps so allow slack
        assert abs(r.mean_predicted - r.mean_observed) < 0.15


def test_requires_fit_and_matching_shapes() -> None:
    cal = IsotonicCalibrator()
    with pytest.raises(RuntimeError):
        cal.predict_proba([0.0])
    with pytest.raises(ValueError, match="same length"):
        cal.fit([0.0] * 40, [1.0] * 39)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        cal.fit([0.0] * 40, [2.0] * 40)
