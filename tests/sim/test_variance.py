import pytest

from regulars.sim.variance import BETWEEN_PAIR_STD_FLOOR, sim_variance_report


def test_constant_scores_are_degenerate() -> None:
    report = sim_variance_report({"ab": [0.8, 0.8, 0.8], "ac": [0.8, 0.8], "bc": [0.8]})
    assert report.degenerate
    assert report.n_pairs == 3
    assert report.between_pair_std == pytest.approx(0.0, abs=1e-12)
    assert report.reason is not None and "between-pair std" in report.reason


def test_near_constant_scores_are_degenerate() -> None:
    report = sim_variance_report({"ab": [0.80, 0.81], "ac": [0.82, 0.83], "bc": [0.81, 0.80]})
    assert report.between_pair_std < BETWEEN_PAIR_STD_FLOOR
    assert report.degenerate


def test_discriminating_sim_is_not_degenerate() -> None:
    report = sim_variance_report(
        {"ab": [0.9, 0.85, 0.95], "ac": [0.2, 0.25, 0.15], "bc": [0.55, 0.5, 0.6]}
    )
    assert not report.degenerate
    assert report.reason is None
    assert report.between_pair_std == pytest.approx(0.2858, abs=1e-3)
    assert report.mean_within_pair_std == pytest.approx(0.0408, abs=1e-3)


def test_seed_noise_larger_than_pair_signal_is_degenerate() -> None:
    # pair means 0.4 / 0.6 / 0.5 (between std ~0.082, above the floor) but every pair swings
    # by 0.8 across seeds (within std 0.4): the seed noise swamps the pair signal.
    report = sim_variance_report({"ab": [0.0, 0.8], "ac": [0.2, 1.0], "bc": [0.1, 0.9]})
    assert report.between_pair_std >= BETWEEN_PAIR_STD_FLOOR
    assert report.mean_within_pair_std == pytest.approx(0.4)
    assert report.degenerate
    assert report.reason is not None and "within-pair" in report.reason


def test_fewer_than_two_pairs_is_degenerate() -> None:
    assert sim_variance_report({}).degenerate
    assert sim_variance_report({"ab": [0.5, 0.9]}).degenerate
    assert sim_variance_report({"ab": [], "ac": [0.3]}).degenerate
