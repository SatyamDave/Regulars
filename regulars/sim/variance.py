"""Variance-before-mean diagnostic (§6 problem 4).

If the spread of scores *between* pairs is near zero, or no larger than the seed-to-seed noise
*within* a pair, the sim is not discriminating between pairs and its mean is meaningless.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean, pstdev

BETWEEN_PAIR_STD_FLOOR = 0.05


@dataclass(frozen=True)
class VarianceReport:
    n_pairs: int
    between_pair_std: float
    mean_within_pair_std: float
    degenerate: bool
    reason: str | None = None


def sim_variance_report(scores_by_pair: Mapping[str, Sequence[float]]) -> VarianceReport:
    """``scores_by_pair`` maps a pair id to its per-(scenario, seed) scores.

    Population standard deviations throughout so a single-sample pair is defined (0.0) rather
    than an error. ``degenerate`` when between-pair std is below ``BETWEEN_PAIR_STD_FLOOR`` or
    below the mean within-pair std.
    """
    pairs = {k: list(v) for k, v in scores_by_pair.items() if len(v) > 0}
    if len(pairs) < 2:
        return VarianceReport(
            n_pairs=len(pairs),
            between_pair_std=0.0,
            mean_within_pair_std=0.0,
            degenerate=True,
            reason="fewer than two pairs with scores",
        )
    means = [fmean(v) for v in pairs.values()]
    between = pstdev(means)
    within = fmean(pstdev(v) for v in pairs.values())
    reason: str | None = None
    if between < BETWEEN_PAIR_STD_FLOOR:
        reason = f"between-pair std {between:.4f} < {BETWEEN_PAIR_STD_FLOOR}"
    elif between < within:
        reason = f"between-pair std {between:.4f} < within-pair std {within:.4f}"
    return VarianceReport(
        n_pairs=len(pairs),
        between_pair_std=between,
        mean_within_pair_std=within,
        degenerate=reason is not None,
        reason=reason,
    )
