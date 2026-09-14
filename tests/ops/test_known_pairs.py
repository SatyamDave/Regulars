"""End-to-end proof of technical-architecture.md §5's build order step 1: feature extractor +
judge v1 must rank known-good pairs above known-bad pairs on real chat exports.

Uses the repo's own hand-written FLOW (follow-ups, converging style, mutual disclosure) and
INTERVIEW (one-sided, flat, no follow-ups) fixtures from ``tests.features.conftest`` -- the same
shapes ``tests/features/test_extract.py`` already proves the real heuristic extractor ranks
correctly feature by feature. Here they are written out as raw chat exports, run through
:mod:`regulars.ops.known_pairs` to build a ``features.jsonl``, and that file is then fed,
unmodified, into the real ``regulars.judge.cli`` known-pairs report.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from regulars.core.features import REAL_ONLY_FEATURES, SHARED_FEATURES
from regulars.core.transcript import Transcript
from regulars.features.extract import ExtractorConfig
from regulars.features.questions import LLMQuestionClassifier
from regulars.judge.cli import run_known_pairs
from regulars.llm.anthropic_client import AnthropicLLM
from regulars.ops.known_pairs import (
    EXAMPLE_GENERIC_MESSAGES,
    PairSpec,
    _extractor_config,
    build_features,
    features_for_pair,
    load_pair_transcript,
    load_pairs_yaml,
    main,
    write_template,
)
from tests.features.conftest import FLOW, INTERVIEW
from tests.features.conftest import A as FLOW_A
from tests.features.conftest import B as FLOW_B

HEURISTIC = ExtractorConfig.heuristic()


def _generic_messages(
    lines: list[tuple[str, str]], a: str, b: str, *, start_ms: int, gap_ms: int
) -> list[dict[str, Any]]:
    speaker = {FLOW_A: a, FLOW_B: b}
    return [
        {"sender": speaker[s], "text": t, "sent_at_ms": start_ms + i * gap_ms}
        for i, (s, t) in enumerate(lines)
    ]


# --- generic JSON shape ------------------------------------------------------------------


def test_generic_shape_builds_a_real_timed_transcript(tmp_path: Path) -> None:
    messages = _generic_messages(FLOW, "priya", "sam", start_ms=0, gap_ms=30_000)
    path = tmp_path / "priya_sam.json"
    path.write_text(json.dumps(messages))
    spec = PairSpec(a="priya", b="sam", label=1, messages=path)

    transcript = load_pair_transcript(spec)

    assert isinstance(transcript, Transcript)
    assert transcript.source == "real"
    assert transcript.has_timing()
    assert set(transcript.participants) == {"priya", "sam"}
    assert [t.text for t in transcript.turns] == [text for _, text in FLOW]
    assert transcript.turns[0].sent_at_ms == 0
    assert transcript.turns[1].sent_at_ms == 30_000


def test_generic_shape_rejects_unknown_sender(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"sender": "zed", "text": "hi", "sent_at_ms": 0}]))
    spec = PairSpec(a="priya", b="sam", label=1, messages=path)
    with pytest.raises(ValueError, match="zed"):
        load_pair_transcript(spec)


def test_generic_shape_rejects_malformed_row(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"sender": "priya", "text": "hi"}]))  # missing sent_at_ms
    spec = PairSpec(a="priya", b="sam", label=1, messages=path)
    with pytest.raises(ValueError, match="malformed"):
        load_pair_transcript(spec)


def test_rejects_empty_messages_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps([]))
    spec = PairSpec(a="priya", b="sam", label=1, messages=path)
    with pytest.raises(ValueError, match="non-empty"):
        load_pair_transcript(spec)


def test_rejects_non_object_entries(tmp_path: Path) -> None:
    path = tmp_path / "strings.json"
    path.write_text(json.dumps(["hi", "there"]))
    spec = PairSpec(a="priya", b="sam", label=1, messages=path)
    with pytest.raises(ValueError, match="message objects"):
        load_pair_transcript(spec)


def test_rejects_bluebubbles_export_with_no_usable_rows(tmp_path: Path) -> None:
    # A tapback *removal* is the only row -- message_row_from_bluebubbles drops it entirely.
    path = tmp_path / "empty_bb.json"
    path.write_text(
        json.dumps(
            [
                {
                    "ROWID": 1,
                    "guid": "m1",
                    "dateCreated": 1_700_000_000_000,
                    "isFromMe": True,
                    "handle": None,
                    "text": "Removed a heart",
                    "associatedMessageType": "3000",
                    "associatedMessageGuid": "p:0/m0",
                }
            ]
        )
    )
    spec = PairSpec(a="priya", b="sam", label=1, messages=path)
    with pytest.raises(ValueError, match="no usable messages"):
        load_pair_transcript(spec)


# --- BlueBubbles shape --------------------------------------------------------------------


def _bluebubbles_export() -> list[dict[str, Any]]:
    """A tiny two-sender BlueBubbles export: priya is the export owner, sam is the contact."""
    return [
        {
            "ROWID": 1,
            "guid": "m1",
            "dateCreated": 1_700_000_000_000,
            "isFromMe": True,
            "handle": None,
            "text": "hey are you around this weekend",
        },
        {
            "ROWID": 2,
            "guid": "m2",
            "dateCreated": 1_700_000_030_000,
            "isFromMe": False,
            "handle": {"address": "+15550001"},
            "text": "yeah what's up",
            "threadOriginatorGuid": "m1",
        },
        {
            "ROWID": 3,
            "guid": "m3",
            "dateCreated": 1_700_000_060_000,
            "isFromMe": True,
            "handle": None,
            "text": "Loved “yeah what's up”",
            "associatedMessageType": 2000,
            "associatedMessageGuid": "p:0/m2",
        },
    ]


def test_bluebubbles_shape_is_detected_and_maps_self_and_other(tmp_path: Path) -> None:
    path = tmp_path / "priya_sam_bb.json"
    path.write_text(json.dumps(_bluebubbles_export()))
    spec = PairSpec(a="priya", b="sam", label=1, messages=path)

    transcript = load_pair_transcript(spec)

    assert transcript.source == "real"
    assert set(transcript.participants) == {"priya", "sam"}
    speakers = [t.speaker_id for t in transcript.turns]
    assert speakers == ["priya", "sam", "priya"]
    # the reply and the tapback both resolved through the guid map
    assert transcript.turns[1].reply_to == 0
    assert transcript.turns[2].is_tapback and transcript.turns[2].tapback_target == 1


# --- pairs.yaml --------------------------------------------------------------------------


def test_load_pairs_yaml_resolves_relative_messages_paths(tmp_path: Path) -> None:
    messages = _generic_messages(FLOW, "a", "b", start_ms=0, gap_ms=1000)
    (tmp_path / "a_b.json").write_text(json.dumps(messages))
    pairs_yaml = tmp_path / "pairs.yaml"
    pairs_yaml.write_text(
        yaml.safe_dump([{"a": "a", "b": "b", "label": 1, "messages": "a_b.json"}])
    )

    specs = load_pairs_yaml(pairs_yaml)

    assert len(specs) == 1
    assert specs[0].messages == tmp_path / "a_b.json"
    assert specs[0].messages.exists()


def test_load_pairs_yaml_rejects_bad_label(tmp_path: Path) -> None:
    pairs_yaml = tmp_path / "pairs.yaml"
    pairs_yaml.write_text(yaml.safe_dump([{"a": "a", "b": "b", "label": 2, "messages": "x.json"}]))
    with pytest.raises(ValueError):
        load_pairs_yaml(pairs_yaml)


def test_load_pairs_yaml_rejects_non_list(tmp_path: Path) -> None:
    pairs_yaml = tmp_path / "pairs.yaml"
    pairs_yaml.write_text(yaml.safe_dump({"a": "a"}))
    with pytest.raises(ValueError, match="expected a YAML list"):
        load_pairs_yaml(pairs_yaml)


def test_load_pairs_yaml_rejects_non_mapping_row(tmp_path: Path) -> None:
    pairs_yaml = tmp_path / "pairs.yaml"
    pairs_yaml.write_text(yaml.safe_dump(["not a mapping"]))
    with pytest.raises(ValueError, match="not a mapping"):
        load_pairs_yaml(pairs_yaml)


def test_load_pairs_yaml_keeps_absolute_messages_path(tmp_path: Path) -> None:
    messages = tmp_path / "abs.json"
    messages.write_text(json.dumps([{"sender": "a", "text": "hi", "sent_at_ms": 0}]))
    pairs_yaml = tmp_path / "pairs.yaml"
    pairs_yaml.write_text(
        yaml.safe_dump([{"a": "a", "b": "b", "label": 1, "messages": str(messages)}])
    )
    specs = load_pairs_yaml(pairs_yaml)
    assert specs[0].messages == messages


def test_build_features_rejects_empty_pairs_file(tmp_path: Path) -> None:
    pairs_yaml = tmp_path / "pairs.yaml"
    pairs_yaml.write_text(yaml.safe_dump([]))
    with pytest.raises(ValueError, match="no pairs"):
        build_features(pairs_yaml, tmp_path / "out.jsonl", "heuristic")


# --- features_for_pair ---------------------------------------------------------------------


def test_features_for_pair_emits_shared_and_real_only_features(tmp_path: Path) -> None:
    messages = _generic_messages(FLOW, "priya", "sam", start_ms=0, gap_ms=30_000)
    path = tmp_path / "priya_sam.json"
    path.write_text(json.dumps(messages))
    spec = PairSpec(a="priya", b="sam", label=1, messages=path)

    row = features_for_pair(spec, HEURISTIC)

    assert row["a"] == "priya"
    assert row["b"] == "sam"
    assert row["label"] == 1
    for features in (row["features_ab"], row["features_ba"]):
        assert set(SHARED_FEATURES) <= set(features)
        assert set(features) <= set(SHARED_FEATURES) | set(REAL_ONLY_FEATURES)
        assert all(isinstance(v, float) for v in features.values())


# --- end-to-end: build -> the real, unmodified known-pairs CLI logic -----------------------


def test_build_features_end_to_end_ranks_known_good_above_known_bad(tmp_path: Path) -> None:
    good_1 = _generic_messages(FLOW, "priya", "sam", start_ms=0, gap_ms=30_000)
    good_2 = _generic_messages(FLOW, "alex", "noor", start_ms=0, gap_ms=45_000)
    bad = _generic_messages(INTERVIEW, "priya", "jordan", start_ms=0, gap_ms=45 * 60_000)

    for name, messages in (
        ("good_1.json", good_1),
        ("good_2.json", good_2),
        ("bad.json", bad),
    ):
        (tmp_path / name).write_text(json.dumps(messages))

    pairs_yaml = tmp_path / "pairs.yaml"
    pairs_yaml.write_text(
        yaml.safe_dump(
            [
                {"a": "priya", "b": "sam", "label": 1, "messages": "good_1.json"},
                {"a": "alex", "b": "noor", "label": 1, "messages": "good_2.json"},
                {"a": "priya", "b": "jordan", "label": 0, "messages": "bad.json"},
            ]
        )
    )
    features_path = tmp_path / "features.jsonl"

    n = build_features(pairs_yaml, features_path, "heuristic")

    assert n == 3
    lines = features_path.read_text().strip().splitlines()
    assert len(lines) == 3
    rows = [json.loads(line) for line in lines]
    assert {row["a"] for row in rows} == {"priya", "alex"}
    for row in rows:
        assert set(row) == {"a", "b", "label", "features_ab", "features_ba"}

    # Fed straight into the real, UNMODIFIED regulars.judge.cli known-pairs logic.
    report = run_known_pairs(features_path)

    assert not report.broken
    assert report.n == 3
    by_label = {(rp.a, rp.b): (rp.score, rp.label) for rp in report.ranking}
    good_scores = [score for (score, label) in by_label.values() if label == 1]
    bad_scores = [score for (score, label) in by_label.values() if label == 0]
    assert len(good_scores) == 2 and len(bad_scores) == 1
    assert min(good_scores) > bad_scores[0]
    # the ranking itself (best first) puts both known-good pairs ahead of the known-bad one
    assert [rp.label for rp in report.ranking] == [1, 1, 0]


# --- template ------------------------------------------------------------------------------


def test_template_writes_a_usable_example(tmp_path: Path) -> None:
    out_dir = tmp_path / "example"

    paths = write_template(out_dir)

    assert all(p.exists() for p in paths)
    specs = load_pairs_yaml(out_dir / "pairs.yaml")
    assert len(specs) == 2
    for spec in specs:
        assert spec.messages.exists()
    # the example messages file itself round-trips through the loader
    first = specs[0]
    transcript = load_pair_transcript(first)
    assert transcript.has_timing()
    assert json.loads((out_dir / "priya_sam.example.json").read_text()) == EXAMPLE_GENERIC_MESSAGES


def test_template_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    write_template(tmp_path)
    with pytest.raises(FileExistsError):
        write_template(tmp_path)
    write_template(tmp_path, force=True)  # does not raise


# --- CLI entrypoint --------------------------------------------------------------------------


def test_extractor_config_heuristic() -> None:
    config = _extractor_config("heuristic")
    assert isinstance(config, ExtractorConfig)


def test_extractor_config_frontier_uses_llm_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = _extractor_config("frontier")
    assert isinstance(config.question_clf, LLMQuestionClassifier)
    assert isinstance(config.question_clf.client, AnthropicLLM)


def test_main_build_writes_features_and_prints_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    messages = _generic_messages(FLOW, "priya", "sam", start_ms=0, gap_ms=30_000)
    (tmp_path / "priya_sam.json").write_text(json.dumps(messages))
    pairs_yaml = tmp_path / "pairs.yaml"
    pairs_yaml.write_text(
        yaml.safe_dump([{"a": "priya", "b": "sam", "label": 1, "messages": "priya_sam.json"}])
    )
    output = tmp_path / "features.jsonl"

    code = main(["build", "--input", str(pairs_yaml), "--output", str(output)])

    assert code == 0
    assert "wrote 1 known pairs" in capsys.readouterr().out
    assert output.exists()


def test_main_template_writes_example_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["template", "--out-dir", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert str(tmp_path / "pairs.yaml") in out
    assert (tmp_path / "pairs.yaml").exists()


def test_module_entrypoint_runs(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "regulars.ops.known_pairs", "template", "--out-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "wrote" in proc.stdout
    assert (tmp_path / "pairs.yaml").exists()
