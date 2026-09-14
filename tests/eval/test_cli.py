import json
from pathlib import Path

import pytest

from regulars.eval.cli import main
from regulars.eval.records import dump_jsonl
from tests.eval.synth import make_dyads, make_retention


def test_cli_gate_on_temp_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    s = make_retention(n_groups_per_arm=20, seed=0)
    dyads = make_dyads(n_groups=10, seed=1, random_share=0.5)
    dump_jsonl(tmp_path / "dyads.jsonl", dyads)
    dump_jsonl(tmp_path / "memberships.jsonl", s.memberships)
    dump_jsonl(tmp_path / "attendance.jsonl", s.attendance)
    dump_jsonl(tmp_path / "rsvps.jsonl", s.rsvps)
    (tmp_path / "yhats.jsonl").write_text(
        "\n".join([json.dumps({"y_hat": v}) for v in s.y_hats_all_pairs[:50]] + ["0.25", "-1.5"])
    )

    rc = main(
        [
            "gate",
            str(tmp_path / "dyads.jsonl"),
            str(tmp_path / "memberships.jsonl"),
            "--attendance",
            str(tmp_path / "attendance.jsonl"),
            "--rsvps",
            str(tmp_path / "rsvps.jsonl"),
            "--yhats",
            str(tmp_path / "yhats.jsonl"),
            "--srm-r2",
            "0.05",
            "--srm-p",
            "0.02",
            "--reps",
            "200",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("# Week 6 gate")
    assert "| yhat_std_all_pairs |" in out and "| 52 |" in out
    assert "| srm_relationship_r2 | 0.050 |" in out
    assert "insufficient_data" not in out


def test_cli_minimal_invocation_falls_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    s = make_retention(n_groups_per_arm=3, seed=0)
    dump_jsonl(tmp_path / "d.jsonl", make_dyads(n_groups=3, seed=0))
    dump_jsonl(tmp_path / "m.jsonl", s.memberships)
    rc = main(["gate", str(tmp_path / "d.jsonl"), str(tmp_path / "m.jsonl"), "--reps", "50"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "| show_rate | — | 0 | insufficient_data |" in out
    assert "--yhats not given" in out


def test_cli_rejects_bad_jsonl(tmp_path: Path) -> None:
    (tmp_path / "d.jsonl").write_text('{"group_id": "g"}\n')
    (tmp_path / "m.jsonl").write_text("")
    with pytest.raises(ValueError, match="d.jsonl:1"):
        main(["gate", str(tmp_path / "d.jsonl"), str(tmp_path / "m.jsonl")])
