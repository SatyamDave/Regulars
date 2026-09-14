"""Judge diagnostics (technical-architecture.md §6 "Problem 4", §5 "Build order").

Run these before looking at any mean. The top technical risk is a sim that collapses to
agreeable: every pair scores high, the spread is zero, and the mean is irrelevant. Watch
variance before mean, then AUC on the known-pairs dev set.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from regulars.core.features import PairFeatures
from regulars.judge.v1 import JudgeV1, pair_score

AUC_BROKEN = 0.6
"""§6: AUC on known pairs under this means the harness is broken, not that everyone is
compatible."""

VARIANCE_FLOOR = 0.05
"""Default std floor for ``variance_check``; below this ``y_hat`` is degenerate."""


@dataclass(frozen=True, slots=True)
class VarianceReport:
    std: float
    degenerate: bool
    iqr: float
    n: int


def variance_check(y_hats: Sequence[float], floor: float = VARIANCE_FLOOR) -> VarianceReport:
    """Spread of ``y_hat`` across pairs. ``degenerate`` when ``std < floor``."""
    arr = np.asarray(y_hats, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("no y_hat values")
    std = float(arr.std(ddof=0))
    q75, q25 = np.percentile(arr, [75, 25])
    return VarianceReport(std=std, degenerate=std < floor, iqr=float(q75 - q25), n=int(arr.size))


def auc_on_known_pairs(y_hats: Sequence[float], labels: Sequence[int]) -> float:
    """ROC AUC of ``y_hats`` against binary ``labels``. Raises if only one class is present."""
    if len(y_hats) != len(labels):
        raise ValueError("y_hats and labels must be the same length")
    classes = set(labels)
    if not classes <= {0, 1}:
        raise ValueError(f"labels must be 0/1, got {sorted(classes)}")
    if len(classes) < 2:
        raise ValueError("AUC needs both a positive and a negative class in labels")
    return float(roc_auc_score(np.asarray(labels), np.asarray(y_hats, dtype=np.float64)))


@dataclass(frozen=True, slots=True)
class KnownPair:
    """One undirected known pair with both directed feature sets and a 0/1 label."""

    a: str
    b: str
    features_ab: PairFeatures
    features_ba: PairFeatures
    label: int


@dataclass(frozen=True, slots=True)
class RankedPair:
    a: str
    b: str
    score: float
    label: int


@dataclass(frozen=True, slots=True)
class KnownPairsReport:
    auc: float
    spearman: float
    ranking: list[RankedPair]
    """Pairs sorted by score, best first."""
    variance: VarianceReport
    n: int

    @property
    def broken(self) -> bool:
        return self.variance.degenerate or self.auc < AUC_BROKEN

    def render(self) -> str:
        degenerate = "  DEGENERATE" if self.variance.degenerate else ""
        broken = f"  BROKEN (< {AUC_BROKEN})" if self.auc < AUC_BROKEN else ""
        lines = [
            f"known pairs: n={self.n}",
            f"variance:    std={self.variance.std:.4f} iqr={self.variance.iqr:.4f}{degenerate}",
            f"AUC:         {self.auc:.3f}{broken}",
            f"spearman:    {self.spearman:.3f}",
            "ranking (best first):",
        ]
        lines.extend(
            f"  {rp.score:+.3f}  label={rp.label}  {rp.a} <-> {rp.b}" for rp in self.ranking
        )
        return "\n".join(lines)


def known_pairs_report(judge: JudgeV1, pairs: Sequence[KnownPair]) -> KnownPairsReport:
    """Score every known pair, rank them, and report AUC, Spearman and variance."""
    if not pairs:
        raise ValueError("no known pairs")
    ranked = [
        RankedPair(
            a=p.a,
            b=p.b,
            score=pair_score(judge.score(p.features_ab), judge.score(p.features_ba)),
            label=p.label,
        )
        for p in pairs
    ]
    scores = [rp.score for rp in ranked]
    labels = [rp.label for rp in ranked]
    variance = variance_check(scores)
    auc = auc_on_known_pairs(scores, labels)
    rho = float(spearmanr(scores, labels).statistic) if variance.std > 0 else float("nan")
    ranked.sort(key=lambda rp: rp.score, reverse=True)
    return KnownPairsReport(auc=auc, spearman=rho, ranking=ranked, variance=variance, n=len(ranked))
