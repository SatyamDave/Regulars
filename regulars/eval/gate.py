"""Week-6 gate report (product-plan §9) plus the §6 success bar.

Product-plan verdicts::

    Round-2 attendance         go ≥ 60%      stop < 30%
    Sim vs reality Spearman    go ≥ 0.35     stop < 0.15   (exploration arm only, clustered SE)
    Show rate                  go ≥ 85%      stop < 70%
    Groups alive at week 6     go > 0        stop = 0

Anything between go and stop is ``fix``. §6 adds: variance of ``y_hat`` across all scored pairs
must be non-degenerate (watch variance before mean), the SRM relationship-variance r² must be
positive and significant, and model-vs-random round-2 retention should show +15pp.

Round-2 attendance is the only number that matters; the overall verdict is the worst metric.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from regulars.eval.correlation import MIN_CLUSTERS, exploration_arm_only, spearman_dyads
from regulars.eval.records import Attendance, LabeledDyad, Membership, Rsvp
from regulars.eval.retention import RetentionResult, round2_retention

Verdict = Literal["go", "fix", "stop", "insufficient_data"]

ROUND2_GO, ROUND2_STOP = 60.0, 30.0
SPEARMAN_GO, SPEARMAN_STOP = 0.35, 0.15
SHOW_GO, SHOW_STOP = 85.0, 70.0
RETENTION_LIFT_GO_PP = 15.0
MIN_YHAT_STD = 0.05
"""Below this the sim is not discriminating (§6 Problem 4). Scale of ``y_hat`` is the judge's."""
MIN_DYADS_FOR_CORR = 30
"""One dinner's worth of directed ratings; below that the Spearman is noise."""
SRM_ALPHA = 0.05

_SEVERITY: dict[Verdict, int] = {"go": 0, "insufficient_data": 1, "fix": 2, "stop": 3}


@dataclass(frozen=True, slots=True)
class GateMetric:
    name: str
    value: float | None
    verdict: Verdict
    n: int
    note: str = ""

    def format_value(self) -> str:
        if self.value is None or math.isnan(self.value):
            return "—"
        return f"{self.value:.3f}" if abs(self.value) < 10 else f"{self.value:.1f}"


@dataclass(frozen=True, slots=True)
class GateReport:
    metrics: tuple[GateMetric, ...]
    retention: RetentionResult | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def verdict(self) -> Verdict:
        worst: Verdict = "go"
        for m in self.metrics:
            if _SEVERITY[m.verdict] > _SEVERITY[worst]:
                worst = m.verdict
        return worst

    def metric(self, name: str) -> GateMetric:
        for m in self.metrics:
            if m.name == name:
                return m
        raise KeyError(name)


def _band(value: float, go: float, stop: float) -> Verdict:
    if value >= go:
        return "go"
    if value < stop:
        return "stop"
    return "fix"


def _round2_metric(ret: RetentionResult) -> GateMetric:
    if ret.n == 0:
        return GateMetric("round2_attendance", None, "insufficient_data", 0, "no round-1 members")
    rate = 100.0 * ret.pooled_rate
    return GateMetric(
        "round2_attendance",
        rate,
        _band(rate, ROUND2_GO, ROUND2_STOP),
        ret.n,
        f"pooled over {ret.n_groups_model + ret.n_groups_random} groups; go ≥ {ROUND2_GO:.0f}%,"
        f" stop < {ROUND2_STOP:.0f}%",
    )


def _retention_lift_metric(ret: RetentionResult) -> GateMetric:
    if ret.n_model == 0 or ret.n_random == 0:
        return GateMetric(
            "retention_lift_model_vs_random",
            None,
            "insufficient_data",
            ret.n,
            f"model n={ret.n_model}, random n={ret.n_random}; both arms required",
        )
    lo, hi = ret.ci95
    ci = f"95% CI [{lo:+.1f}, {hi:+.1f}]pp" if not math.isnan(lo) else "CI needs ≥2 groups per arm"
    verdict: Verdict = "go" if ret.diff_pp >= RETENTION_LIFT_GO_PP else "fix"
    return GateMetric(
        "retention_lift_model_vs_random",
        ret.diff_pp,
        verdict,
        ret.n,
        f"model {100 * ret.model_rate:.1f}% vs random {100 * ret.random_rate:.1f}%; {ci};"
        f" bar +{RETENTION_LIFT_GO_PP:.0f}pp",
    )


def _spearman_metric(dyads: Sequence[LabeledDyad], reps: int, seed: int) -> GateMetric:
    arm = exploration_arm_only(dyads)
    n_groups = len({d.group_id for d in arm})
    excluded = len(dyads) - len(arm)
    if len(arm) < MIN_DYADS_FOR_CORR or n_groups < MIN_CLUSTERS:
        return GateMetric(
            "spearman_exploration_arm",
            None,
            "insufficient_data",
            len(arm),
            f"{len(arm)} exploration-arm dyads in {n_groups} groups (need ≥{MIN_DYADS_FOR_CORR}"
            f" in ≥{MIN_CLUSTERS}); {excluded} model-arm dyads excluded",
        )
    try:
        res = spearman_dyads(arm, reps=reps, seed=seed)
    except ValueError as exc:
        return GateMetric(
            "spearman_exploration_arm", None, "insufficient_data", len(arm), f"undefined: {exc}"
        )
    return GateMetric(
        "spearman_exploration_arm",
        res.rho,
        _band(res.rho, SPEARMAN_GO, SPEARMAN_STOP),
        res.n,
        f"clustered SE {res.se:.3f}, 95% CI [{res.ci95[0]:.2f}, {res.ci95[1]:.2f}];"
        f" {res.n_eff_note}; {excluded} model-arm dyads excluded",
    )


def _show_rate_metric(rsvps: Sequence[Rsvp], attendance: Sequence[Attendance]) -> GateMetric:
    """Attended / committed RSVPs, over plans that have any attendance recorded."""
    recorded_plans = {a.plan_id for a in attendance}
    attended = {(a.plan_id, a.user_id) for a in attendance if a.attended}
    committed = [
        (r.plan_id, r.user_id) for r in rsvps if r.committed and r.plan_id in recorded_plans
    ]
    if not committed:
        return GateMetric(
            "show_rate",
            None,
            "insufficient_data",
            0,
            f"{len(rsvps)} rsvps, {len(recorded_plans)} plans with attendance; no committed"
            " rsvps on recorded plans",
        )
    shows = sum(1 for key in committed if key in attended)
    rate = 100.0 * shows / len(committed)
    return GateMetric(
        "show_rate",
        rate,
        _band(rate, SHOW_GO, SHOW_STOP),
        len(committed),
        f"{shows}/{len(committed)} committed rsvps attended across {len(recorded_plans)} plans;"
        f" go ≥ {SHOW_GO:.0f}%, stop < {SHOW_STOP:.0f}%",
    )


def _groups_alive_metric(memberships: Sequence[Membership]) -> GateMetric:
    active: dict[str, int] = defaultdict(int)
    all_groups: set[str] = set()
    for m in memberships:
        all_groups.add(m.group_id)
        if m.active:
            active[m.group_id] += 1
    if not all_groups:
        return GateMetric("groups_alive", None, "insufficient_data", 0, "no memberships")
    alive = len(active)
    verdict: Verdict = "go" if alive > 0 else "stop"
    return GateMetric(
        "groups_alive",
        float(alive),
        verdict,
        len(all_groups),
        f"{alive}/{len(all_groups)} groups have ≥1 member with left_round unset",
    )


def _variance_metric(y_hats: Sequence[float]) -> GateMetric:
    arr = np.asarray(y_hats, dtype=np.float64)
    if arr.size < 2:
        return GateMetric(
            "yhat_std_all_pairs", None, "insufficient_data", int(arr.size), "<2 pairs"
        )
    std = float(np.std(arr, ddof=1))
    verdict: Verdict = "go" if std >= MIN_YHAT_STD else "fix"
    return GateMetric(
        "yhat_std_all_pairs",
        std,
        verdict,
        int(arr.size),
        f"mean {arr.mean():.3f}, IQR {np.percentile(arr, 75) - np.percentile(arr, 25):.3f};"
        f" degenerate below {MIN_YHAT_STD}: sim is not discriminating (watch variance first)",
    )


def _srm_metric(r2: float | None, p: float | None) -> GateMetric:
    if r2 is None:
        return GateMetric(
            "srm_relationship_r2",
            None,
            "insufficient_data",
            0,
            "SRM decomposition not supplied (needs ~1,000 labeled directed dyads)",
        )
    if p is None:
        verdict: Verdict = "go" if r2 > 0 else "fix"
        note = "no p-value supplied; sign only"
    else:
        verdict = "go" if (r2 > 0 and p < SRM_ALPHA) else "fix"
        note = f"p = {p:.3g}; need r² > 0 and p < {SRM_ALPHA}"
    return GateMetric("srm_relationship_r2", r2, verdict, 0, note + "; charisma-detector check")


def week6_gate(
    dyads: Sequence[LabeledDyad],
    memberships: Sequence[Membership],
    attendance: Sequence[Attendance],
    rsvps: Sequence[Rsvp],
    y_hats_all_pairs: Sequence[float],
    srm_relationship_r2: float | None = None,
    srm_relationship_p: float | None = None,
    reps: int = 2000,
    seed: int = 0,
) -> GateReport:
    retention = round2_retention(memberships, attendance, reps=reps, seed=seed)
    metrics = (
        _round2_metric(retention),
        _spearman_metric(dyads, reps, seed),
        _show_rate_metric(rsvps, attendance),
        _groups_alive_metric(memberships),
        _variance_metric(y_hats_all_pairs),
        _srm_metric(srm_relationship_r2, srm_relationship_p),
        _retention_lift_metric(retention),
    )
    notes = (
        "Round-2 attendance is the only number that matters (product-plan §9).",
        "Spearman is exploration arm only with cluster-bootstrap SEs (§6 Problems 1 and 3).",
    )
    return GateReport(metrics=metrics, retention=retention, notes=notes)


def render_markdown(report: GateReport) -> str:
    lines = [
        "# Week 6 gate",
        "",
        f"**Overall verdict: {report.verdict.upper()}**",
        "",
        "| Metric | Value | n | Verdict | Note |",
        "|---|---:|---:|---|---|",
    ]
    for m in report.metrics:
        note = m.note.replace("|", "\\|")
        lines.append(f"| {m.name} | {m.format_value()} | {m.n} | {m.verdict} | {note} |")
    lines.append("")
    lines.extend(f"- {n}" for n in report.notes)
    lines.append("")
    return "\n".join(lines)
