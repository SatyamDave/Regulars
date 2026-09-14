"""The judge (technical-architecture.md §5, §6 Problem 4).

v1 linear judge over z-scored ``PairFeatures``, isotonic calibration, Kenny's Social
Relations Model for round-robin ratings, and the variance/AUC diagnostics that must be run
before any mean is believed.
"""

from regulars.judge.calibration import MIN_CALIBRATION_N, IsotonicCalibrator, ReliabilityRow
from regulars.judge.diagnostics import (
    AUC_BROKEN,
    KnownPair,
    KnownPairsReport,
    RankedPair,
    VarianceReport,
    auc_on_known_pairs,
    known_pairs_report,
    variance_check,
)
from regulars.judge.srm import (
    DirectedRating,
    GroupEffects,
    SRMResult,
    fit_srm,
    group_effects,
    relationship_residuals,
    variance_explained_by,
)
from regulars.judge.v1 import (
    DEFAULT_VERSION,
    DEFAULT_WEIGHTS,
    JudgeOutput,
    JudgeV1,
    ZScaler,
    pair_score,
)

__all__ = [
    "AUC_BROKEN",
    "DEFAULT_VERSION",
    "DEFAULT_WEIGHTS",
    "MIN_CALIBRATION_N",
    "DirectedRating",
    "GroupEffects",
    "IsotonicCalibrator",
    "JudgeOutput",
    "JudgeV1",
    "KnownPair",
    "KnownPairsReport",
    "RankedPair",
    "ReliabilityRow",
    "SRMResult",
    "VarianceReport",
    "ZScaler",
    "auc_on_known_pairs",
    "fit_srm",
    "group_effects",
    "known_pairs_report",
    "pair_score",
    "relationship_residuals",
    "variance_check",
    "variance_explained_by",
]
