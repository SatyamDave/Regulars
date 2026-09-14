"""Evaluation (§6): person-grouped CV, clustered SEs, mixed models, week-6 gate report.

Consumes plain records (:mod:`regulars.eval.records`); never imports the judge or the sim.
"""

from regulars.eval.correlation import CorrResult, exploration_arm_only, spearman_clustered
from regulars.eval.gate import GateMetric, GateReport, render_markdown, week6_gate
from regulars.eval.mixed import MixedResult, fit_mixed
from regulars.eval.power import effective_n, n_needed, power_for_rho
from regulars.eval.records import Attendance, LabeledDyad, Membership, Rsvp, load_dyads
from regulars.eval.retention import RetentionResult, round2_retention
from regulars.eval.splits import FoldSplit, assert_no_leakage, person_grouped_folds

__all__ = [
    "Attendance",
    "CorrResult",
    "FoldSplit",
    "GateMetric",
    "GateReport",
    "LabeledDyad",
    "Membership",
    "MixedResult",
    "RetentionResult",
    "Rsvp",
    "assert_no_leakage",
    "effective_n",
    "exploration_arm_only",
    "fit_mixed",
    "load_dyads",
    "n_needed",
    "person_grouped_folds",
    "power_for_rho",
    "render_markdown",
    "round2_retention",
    "spearman_clustered",
    "week6_gate",
]
