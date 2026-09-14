import pytest

from regulars.eval.mixed import fit_mixed
from tests.eval.synth import make_dyads


def test_mixed_model_recovers_positive_beta_and_converges() -> None:
    dyads = make_dyads(n_groups=10, seed=1, beta=0.8)
    res = fit_mixed(dyads)
    assert res.converged
    assert res.n == 300
    assert res.beta_yhat > 0.4  # attenuated by rounding onto 1..7, still clearly positive
    assert res.p < 1e-6
    assert res.se > 0
    # every planted component is non-trivial
    assert res.var_rater > 0.05
    assert res.var_ratee > 0.05
    assert res.var_group > 0.05
    assert res.var_resid > 0.1
    assert res.share("resid") == pytest.approx(res.var_resid / res.total_var)


def test_mixed_model_no_signal_gives_small_beta() -> None:
    dyads = make_dyads(n_groups=10, seed=5, beta=0.0)
    res = fit_mixed(dyads)
    assert abs(res.beta_yhat) < 0.2
    assert res.p > 0.01


def test_mixed_model_rejects_tiny_or_constant_input() -> None:
    dyads = make_dyads(n_groups=1, seed=0)
    with pytest.raises(ValueError, match="at least"):
        fit_mixed(dyads[:5])
    flat = [d.model_copy(update={"y_hat": 0.0}) for d in dyads]
    with pytest.raises(ValueError, match="constant"):
        fit_mixed(flat)
