from regulars.eval.gate import (
    MIN_YHAT_STD,
    GateMetric,
    GateReport,
    render_markdown,
    week6_gate,
)
from regulars.eval.records import Attendance, Membership, Rsvp
from tests.eval.synth import make_dyads, make_retention

METRIC_NAMES = {
    "round2_attendance",
    "spearman_exploration_arm",
    "show_rate",
    "groups_alive",
    "yhat_std_all_pairs",
    "srm_relationship_r2",
    "retention_lift_model_vs_random",
}


def test_empty_input_reports_insufficient_data_everywhere() -> None:
    report = week6_gate([], [], [], [], [])
    assert {m.name for m in report.metrics} == METRIC_NAMES
    assert all(m.verdict == "insufficient_data" for m in report.metrics)
    assert report.verdict == "insufficient_data"
    md = render_markdown(report)
    assert "INSUFFICIENT_DATA" in md
    assert md.count("| insufficient_data |") == len(METRIC_NAMES)
    assert "| — |" in md


def test_healthy_synthetic_week_goes_except_missing_srm() -> None:
    s = make_retention(n_groups_per_arm=50, p_model=0.80, p_random=0.62, seed=2)
    dyads = make_dyads(n_groups=20, seed=1, beta=0.8, random_share=0.5)
    report = week6_gate(
        dyads, s.memberships, s.attendance, s.rsvps, s.y_hats_all_pairs, reps=300, seed=0
    )
    by = {m.name: m for m in report.metrics}
    assert by["round2_attendance"].verdict == "go"
    assert by["round2_attendance"].value is not None and by["round2_attendance"].value >= 60
    assert by["spearman_exploration_arm"].verdict == "go"
    assert by["spearman_exploration_arm"].n == 10 * 30  # model arm excluded
    assert "300 model-arm dyads excluded" in by["spearman_exploration_arm"].note
    assert by["show_rate"].verdict == "go"
    assert by["groups_alive"].verdict == "go"
    assert by["yhat_std_all_pairs"].verdict == "go"
    assert by["srm_relationship_r2"].verdict == "insufficient_data"
    assert by["retention_lift_model_vs_random"].verdict == "go"
    assert report.verdict == "insufficient_data"  # the §6 bar is not certified without SRM

    with_srm = week6_gate(
        dyads,
        s.memberships,
        s.attendance,
        s.rsvps,
        s.y_hats_all_pairs,
        srm_relationship_r2=0.08,
        srm_relationship_p=0.01,
        reps=300,
    )
    assert with_srm.metric("srm_relationship_r2").verdict == "go"
    assert with_srm.verdict == "go"
    md = render_markdown(with_srm)
    assert "**Overall verdict: GO**" in md
    assert "exploration arm only" in md


def test_stop_and_fix_verdicts() -> None:
    # dead product: nobody comes back, all groups closed, sim is flat, srm negative
    memberships = [
        Membership(
            group_id=f"g{g}",
            user_id=f"u{g}{i}",
            joined_round=1,
            left_round=2,
            exit_reason="quit",
            assembly_method="random" if g % 2 else "model",
        )
        for g in range(6)
        for i in range(6)
    ]
    attendance = [
        Attendance(
            plan_id=f"g{g}-1", group_id=f"g{g}", user_id=f"u{g}{i}", round_index=1, attended=i < 4
        )
        for g in range(6)
        for i in range(6)
    ]
    rsvps = [
        Rsvp(plan_id=f"g{g}-1", user_id=f"u{g}{i}", state="yes") for g in range(6) for i in range(6)
    ]
    dyads = make_dyads(n_groups=10, seed=9, beta=0.0)  # no signal
    report = week6_gate(
        dyads,
        memberships,
        attendance,
        rsvps,
        [0.5] * 100,
        srm_relationship_r2=-0.01,
        srm_relationship_p=0.5,
        reps=200,
    )
    by = {m.name: m for m in report.metrics}
    assert by["round2_attendance"].verdict == "stop"
    assert by["groups_alive"].verdict == "stop"
    assert by["show_rate"].value is not None and abs(by["show_rate"].value - 66.7) < 0.1
    assert by["show_rate"].verdict == "stop"
    assert by["spearman_exploration_arm"].verdict in {"stop", "fix"}
    assert by["yhat_std_all_pairs"].verdict == "fix"
    assert by["srm_relationship_r2"].verdict == "fix"
    assert by["retention_lift_model_vs_random"].verdict == "fix"
    assert report.verdict == "stop"
    md = render_markdown(report)
    assert "STOP" in md
    for verdict in ("stop", "fix"):
        assert f"| {verdict} |" in md


def test_fix_band_and_severity_order() -> None:
    m = [
        GateMetric("a", 50.0, "go", 10),
        GateMetric("b", 50.0, "fix", 10, "between go and stop"),
    ]
    assert GateReport(metrics=tuple(m)).verdict == "fix"
    m.append(GateMetric("c", None, "insufficient_data", 0))
    assert GateReport(metrics=tuple(m)).verdict == "fix"
    m.append(GateMetric("d", 0.0, "stop", 3))
    assert GateReport(metrics=tuple(m)).verdict == "stop"
    assert GateReport(metrics=(GateMetric("x", MIN_YHAT_STD, "go", 2),)).verdict == "go"
    md = render_markdown(GateReport(metrics=tuple(m)))
    assert "| b | 50.0 | 10 | fix | between go and stop |" in md
