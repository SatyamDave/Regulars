import json
import subprocess
import sys
from pathlib import Path

import pytest

from regulars.judge.cli import load_known_pairs, main, run_known_pairs
from regulars.judge.v1 import DEFAULT_WEIGHTS, JudgeV1, ZScaler

ROWS = [
    {
        "a": "priya",
        "b": "sam",
        "features_ab": {"follow_up_rate": 0.5, "question_gini": 0.1, "latency_norm_median": -0.4},
        "features_ba": {"follow_up_rate": 0.4, "question_gini": 0.1, "latency_norm_median": -0.2},
        "label": 1,
    },
    {
        "a": "lee",
        "b": "kim",
        "features_ab": {"follow_up_rate": 0.35, "question_gini": 0.3},
        "features_ba": {"follow_up_rate": 0.3, "question_gini": 0.3},
        "label": 1,
    },
    {
        "a": "ana",
        "b": "bo",
        "features_ab": {"follow_up_rate": 0.1, "question_gini": 0.7, "latency_norm_median": 0.5},
        "features_ba": {"follow_up_rate": 0.05, "question_gini": 0.7, "latency_norm_median": 0.3},
        "label": 0,
    },
    {
        "a": "cy",
        "b": "di",
        "features_ab": {"follow_up_rate": 0.0, "question_gini": 0.9},
        "features_ba": {"follow_up_rate": 0.1, "question_gini": 0.9},
        "label": 0,
    },
]


@pytest.fixture
def jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "features.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n\n")
    return path


def test_load_known_pairs(jsonl: Path) -> None:
    pairs = load_known_pairs(jsonl)
    assert [p.a for p in pairs] == ["priya", "lee", "ana", "cy"]
    assert pairs[0].features_ba.user_a == "sam"
    assert pairs[0].features_ab.features["latency_norm_median"] == -0.4


def test_load_rejects_malformed_rows(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"a": "x", "b": "y", "features_ab": {}, "label": 1}\n')
    with pytest.raises(ValueError, match="malformed"):
        load_known_pairs(bad)
    bad.write_text('{"a": "x", "b": "y", "features_ab": {}, "features_ba": {}, "label": 3}\n')
    with pytest.raises(ValueError, match="label must be 0 or 1"):
        load_known_pairs(bad)
    bad.write_text("not json\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_known_pairs(bad)


def test_main_prints_report_and_exits_zero(jsonl: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["known-pairs", str(jsonl)])
    out = capsys.readouterr().out
    assert code == 0
    assert "known pairs: n=4" in out
    assert "AUC:         1.000" in out
    assert "priya <-> sam" in out


def test_main_with_serialised_judge(jsonl: Path, tmp_path: Path) -> None:
    pairs = load_known_pairs(jsonl)
    judge = JudgeV1(
        weights=dict(DEFAULT_WEIGHTS),
        scaler=ZScaler.fit([pf for p in pairs for pf in (p.features_ab, p.features_ba)]),
        version="pinned",
    )
    judge_path = tmp_path / "judge.json"
    judge_path.write_text(judge.to_json())
    report = run_known_pairs(jsonl, judge_path)
    assert report.auc == 1.0


def test_module_entrypoint_runs(jsonl: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "regulars.judge.cli", "known-pairs", str(jsonl)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ranking (best first):" in proc.stdout
