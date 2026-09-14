import pytest

from regulars.eval.power import design_effect, effective_n, n_needed, power_for_rho


def test_n_needed_for_rho_035_matches_fisher_z() -> None:
    n = n_needed(0.35, 0.8)
    assert 60 <= n <= 63  # ((1.96 + 0.84) / atanh(0.35))^2 + 3 = 61.8
    assert n_needed(0.35, 0.8, design_effect=2.0) == pytest.approx(2 * n, abs=2)
    assert n_needed(0.35, 0.8, design_effect=3.0) > n_needed(0.35, 0.8, design_effect=2.0)
    assert n_needed(0.15, 0.8) > 300  # the "stop" threshold needs far more data
    assert n_needed(-0.35, 0.8) == n


def test_power_is_monotone_and_consistent_with_n_needed() -> None:
    n = n_needed(0.35, 0.8)
    assert power_for_rho(0.35, n) >= 0.8
    assert power_for_rho(0.35, n - 5) < 0.8
    assert power_for_rho(0.35, 300) > 0.99
    assert power_for_rho(0.35, 300, design_effect=3.0) < power_for_rho(0.35, 300)
    assert power_for_rho(0.35, 3) == 0.0
    assert power_for_rho(0.0, 300) == pytest.approx(0.05, abs=0.01)


def test_effective_n_kish() -> None:
    # 30 directed ratings per dinner, ICC 0.2 -> design effect 1 + 29 * 0.2 = 6.8
    assert design_effect(30, 0.2) == pytest.approx(6.8)
    assert effective_n(1800, 30, 0.2) == pytest.approx(1800 / 6.8)
    assert effective_n(1800, 30, 0.0) == 1800
    assert effective_n(1800, 1, 0.9) == 1800


def test_argument_validation() -> None:
    with pytest.raises(ValueError):
        n_needed(1.0, 0.8)
    with pytest.raises(ValueError):
        n_needed(0.35, 1.0)
    with pytest.raises(ValueError):
        power_for_rho(0.35, 100, design_effect=0.5)
    with pytest.raises(ValueError):
        effective_n(100, 6, 1.5)
