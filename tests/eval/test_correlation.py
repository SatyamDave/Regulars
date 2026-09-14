import numpy as np
import pytest

from regulars.eval.correlation import exploration_arm_only, spearman_clustered, spearman_dyads
from tests.eval.synth import make_dyads


def test_recovers_planted_rho_and_clustered_se_exceeds_naive() -> None:
    # 60 dinners, beta tuned so the population Spearman sits near the 0.35 bar, with a
    # group-level component in both y_hat and the rating so observations cluster by table.
    dyads = make_dyads(n_groups=60, seed=1, beta=0.6)
    res = spearman_dyads(dyads, reps=2000, seed=0)
    assert res.rho == pytest.approx(0.35, abs=0.05)
    assert res.n == 1800
    assert res.n_clusters == 60
    assert res.n_raters == 360 and res.n_ratees == 360
    assert res.ci95[0] < res.rho < res.ci95[1]
    # the point of clustered SEs (§6 Problem 3): i.i.d. overstates precision
    assert res.se > res.naive_se * 1.3
    assert res.design_effect > 1.5
    assert "design effect" in res.n_eff_note


def test_iid_data_gives_similar_clustered_and_naive_se() -> None:
    rng = np.random.default_rng(0)
    n = 1200
    x = rng.normal(size=n)
    y = 0.4 * x + rng.normal(size=n)
    clusters = [f"c{i % 40}" for i in range(n)]  # labels carry no structure
    res = spearman_clustered(x, y, clusters, reps=1000, seed=0)
    assert res.se == pytest.approx(res.naive_se, rel=0.35)


def test_exploration_filter_excludes_model_arm() -> None:
    dyads = make_dyads(n_groups=10, seed=0, random_share=0.3)
    arm = exploration_arm_only(dyads)
    assert arm and all(d.assembly_method == "random" for d in arm)
    assert len(arm) == 3 * 30
    assert len(dyads) - len(arm) == 7 * 30


def test_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError, match="clusters"):
        spearman_clustered([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], ["a", "a", "a"])
    with pytest.raises(ValueError, match="constant"):
        spearman_clustered([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0], ["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="same length"):
        spearman_clustered([1.0, 2.0], [1.0, 2.0, 3.0], ["a", "b"])
