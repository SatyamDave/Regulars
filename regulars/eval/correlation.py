"""Spearman ρ with cluster-bootstrap standard errors (§6 Problem 3).

Directed ratings are clustered within group (six people, thirty ratings) and within rater and
ratee. An i.i.d. SE overstates precision. We resample whole clusters with replacement and take
the bootstrap SD as the SE; ``naive_se`` is the i.i.d. bootstrap over dyads for comparison.

The headline number is computed on the exploration arm only (§6 Problem 1):
:func:`exploration_arm_only` keeps ``assembly_method == 'random'``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import spearmanr

from regulars.eval.records import LabeledDyad

MIN_CLUSTERS = 2
"""Fewer clusters than this cannot be bootstrapped; :func:`spearman_clustered` refuses."""


@dataclass(frozen=True, slots=True)
class CorrResult:
    rho: float
    se: float
    ci95: tuple[float, float]
    n: int
    n_clusters: int
    n_raters: int
    n_ratees: int
    naive_se: float
    n_eff_note: str

    @property
    def design_effect(self) -> float:
        """Variance inflation of the clustered SE over the i.i.d. SE (≥ 1 means clustering bit)."""
        if self.naive_se <= 0.0:
            return math.inf
        return (self.se / self.naive_se) ** 2


def exploration_arm_only(dyads: Sequence[LabeledDyad]) -> list[LabeledDyad]:
    """The unbiased evaluation set: groups the model did not pick."""
    return [d for d in dyads if d.assembly_method == "random"]


def _spearman(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    if x.size < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return math.nan
    return float(spearmanr(x, y).statistic)


def spearman_clustered(
    y_hat: ArrayLike,
    y: ArrayLike,
    clusters: Sequence[str] | ArrayLike,
    reps: int = 2000,
    seed: int = 0,
    raters: Sequence[str] | None = None,
    ratees: Sequence[str] | None = None,
) -> CorrResult:
    """Spearman ρ of ``y_hat`` vs ``y`` with SE from a cluster bootstrap over ``clusters``.

    ``clusters`` is normally the ``group_id`` of each dyad. ``raters`` / ``ratees`` are optional
    and only feed the reported cluster counts (they are not resampled: a person-level bootstrap
    on top of the group bootstrap double-counts; the group is the coarsest, dominant cluster).
    """
    x_arr = np.asarray(y_hat, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    c_arr = np.asarray(clusters)
    if not (x_arr.shape == y_arr.shape == c_arr.shape) or x_arr.ndim != 1:
        raise ValueError("y_hat, y and clusters must be 1-d and the same length")
    n = int(x_arr.size)
    if n < 3:
        raise ValueError("need at least 3 dyads")
    if reps < 1:
        raise ValueError("reps must be >= 1")

    labels, inverse = np.unique(c_arr, return_inverse=True)
    n_clusters = int(labels.size)
    if n_clusters < MIN_CLUSTERS:
        raise ValueError(f"need at least {MIN_CLUSTERS} clusters, got {n_clusters}")
    members: list[NDArray[np.intp]] = [
        np.flatnonzero(inverse == k).astype(np.intp) for k in range(n_clusters)
    ]

    rho = _spearman(x_arr, y_arr)
    if math.isnan(rho):
        raise ValueError("Spearman undefined: y_hat or y is constant")

    rng = np.random.default_rng(seed)
    clustered = np.empty(reps, dtype=np.float64)
    naive = np.empty(reps, dtype=np.float64)
    for r in range(reps):
        picked = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([members[k] for k in picked])
        clustered[r] = _spearman(x_arr[idx], y_arr[idx])
        iid = rng.integers(0, n, size=n)
        naive[r] = _spearman(x_arr[iid], y_arr[iid])

    clustered = clustered[~np.isnan(clustered)]
    naive = naive[~np.isnan(naive)]
    if clustered.size < 2 or naive.size < 2:
        raise ValueError("bootstrap degenerate: too many constant resamples")
    se = float(np.std(clustered, ddof=1))
    lo, hi = (float(v) for v in np.percentile(clustered, [2.5, 97.5]))
    naive_se = float(np.std(naive, ddof=1))

    n_raters = len(set(raters)) if raters is not None else 0
    n_ratees = len(set(ratees)) if ratees is not None else 0
    deff = (se / naive_se) ** 2 if naive_se > 0 else math.inf
    note = (
        f"{n} directed dyads in {n_clusters} groups"
        + (f", {n_raters} raters, {n_ratees} ratees" if raters is not None else "")
        + f"; cluster-bootstrap SE {se:.3f} vs i.i.d. {naive_se:.3f}"
        + f" (design effect {deff:.2f}, n_eff ≈ {n / deff:.0f})"
    )
    return CorrResult(
        rho=rho,
        se=se,
        ci95=(lo, hi),
        n=n,
        n_clusters=n_clusters,
        n_raters=n_raters,
        n_ratees=n_ratees,
        naive_se=naive_se,
        n_eff_note=note,
    )


def spearman_dyads(
    dyads: Sequence[LabeledDyad],
    reps: int = 2000,
    seed: int = 0,
    use_calibrated: bool = False,
) -> CorrResult:
    """Convenience wrapper: clustered Spearman of the judge score vs the real rating."""
    if use_calibrated and any(d.y_hat_calibrated is None for d in dyads):
        raise ValueError("use_calibrated requires y_hat_calibrated on every dyad")
    y_hat = [
        (d.y_hat_calibrated if use_calibrated and d.y_hat_calibrated is not None else d.y_hat)
        for d in dyads
    ]
    return spearman_clustered(
        y_hat,
        [d.outcome for d in dyads],
        [d.group_id for d in dyads],
        reps=reps,
        seed=seed,
        raters=[d.rater for d in dyads],
        ratees=[d.ratee for d in dyads],
    )
