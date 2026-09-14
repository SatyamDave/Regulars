from __future__ import annotations

import json
from pathlib import Path

import pytest

from regulars.core import FakeLLM
from regulars.ops.persona_diagnostic import (
    Label,
    Person,
    build_personas,
    cmd_personas,
    cmd_template,
    cmd_variance,
    load_people_file,
    load_personas,
    render_report,
    run_variance_diagnostic,
    write_personas,
)
from regulars.sim.harness import SimConfig
from regulars.sim.scenarios import COLD_OPEN, DEAD_AIR

PERSONA_JSON = json.dumps(
    {
        "conversational": {"opener_style": "direct", "question_tendency": "high"},
        "content": {"unprompted_topics": ["climbing"]},
        "social": {"friction_points": ["being interrupted"], "drains": ["small talk"]},
    }
)


def persona_json(tag: str) -> str:
    """A distinct PersonaSpec JSON per tag, so the sim cache's content-hash key never collides
    across test personas (real people never share byte-identical specs; two personas built from
    the same fixture string in a test would, which is a test bug, not a product one)."""
    return json.dumps(
        {
            "conversational": {"opener_style": f"direct-{tag}", "question_tendency": "high"},
            "content": {"unprompted_topics": [f"topic-{tag}"]},
            "social": {"friction_points": [f"friction-{tag}"], "drains": [f"drain-{tag}"]},
        }
    )


def make_llm(replies: list[str]) -> FakeLLM:
    return FakeLLM(replies=list(replies))


async def test_build_personas_extracts_and_populates_negative_space() -> None:
    people = [
        Person(id="priya", answers=["a", "b", "c"]),
        Person(id="sam", answers=["x", "y", "z"]),
    ]
    llm = make_llm([PERSONA_JSON, PERSONA_JSON])
    personas = await build_personas(people, llm, concurrency=2)
    assert set(personas) == {"priya", "sam"}
    assert personas["priya"].has_negative_space()


def test_write_and_load_personas_round_trip(tmp_path: Path) -> None:
    from regulars.core import PersonaSpec

    spec = PersonaSpec.model_validate_json(PERSONA_JSON)
    write_personas({"priya": spec, "sam": spec}, tmp_path)
    loaded = load_personas(tmp_path)
    assert set(loaded) == {"priya", "sam"}
    assert loaded["priya"].social.friction_points == ["being interrupted"]


def test_load_people_file_parses_answers_and_labels(tmp_path: Path) -> None:
    p = tmp_path / "people.yaml"
    p.write_text(
        """
people:
  - id: priya
    answers: ["a", "b", "c"]
  - id: sam
    answers: ["x", "y", "z"]
labels:
  - {a: priya, b: sam, would_see_again: true}
"""
    )
    pf = load_people_file(p)
    assert [person.id for person in pf.people] == ["priya", "sam"]
    assert pf.labels == [Label(a="priya", b="sam", would_see_again=True)]


def test_person_requires_exactly_three_answers() -> None:
    with pytest.raises(ValueError, match="at least 3|at most 3"):
        Person(id="x", answers=["a", "b"])


async def test_variance_diagnostic_flags_degenerate_when_transcripts_are_identical(
    tmp_path: Path,
) -> None:
    from regulars.core import PersonaSpec

    personas = {
        "priya": PersonaSpec.model_validate_json(persona_json("priya")),
        "sam": PersonaSpec.model_validate_json(persona_json("sam")),
        "jordan": PersonaSpec.model_validate_json(persona_json("jordan")),
    }
    # Every simulated turn is one of the same three flat lines regardless of who's "speaking" ->
    # transcripts across all pairs end up feature-identical -> zero between-pair spread expected.
    llm = FakeLLM(
        responder=lambda system, messages: ["totally", "yeah exactly", "for sure"][
            len(messages) % 3
        ]
    )
    result = await run_variance_diagnostic(
        personas,
        llm,
        config=SimConfig(turns=4, seeds=(0,)),
        scenarios=[COLD_OPEN],
    )
    assert result.report.n_pairs == 3
    assert result.report.degenerate is True
    assert len(result.ranked_pairs) == 3


async def test_variance_diagnostic_reports_auc_when_labels_given() -> None:
    from regulars.core import PersonaSpec

    personas = {
        "priya": PersonaSpec.model_validate_json(persona_json("priya")),
        "sam": PersonaSpec.model_validate_json(persona_json("sam")),
    }
    llm = FakeLLM(responder=lambda system, messages: "sure, tell me more" * (len(messages) % 3 + 1))
    labels = [Label(a="priya", b="sam", would_see_again=True)]
    result = await run_variance_diagnostic(
        personas,
        llm,
        config=SimConfig(turns=4, seeds=(0,)),
        scenarios=[COLD_OPEN],
        labels=labels,
    )
    # A single labeled pair can't produce two classes -> auc_note explains why, not a crash.
    assert result.auc is None
    assert result.auc_note is not None


def test_render_report_includes_verdict_for_both_outcomes() -> None:
    from regulars.sim.variance import VarianceReport

    degenerate = VarianceReport(
        n_pairs=3,
        between_pair_std=0.01,
        mean_within_pair_std=0.02,
        degenerate=True,
        reason="too low",
    )
    ok = VarianceReport(n_pairs=3, between_pair_std=0.5, mean_within_pair_std=0.1, degenerate=False)
    from regulars.ops.persona_diagnostic import VarianceRunResult

    fail_text = render_report(
        VarianceRunResult(
            report=degenerate, ranked_pairs=[("a", "b", 0.1)], auc=None, auc_note=None
        )
    )
    pass_text = render_report(
        VarianceRunResult(report=ok, ranked_pairs=[("a", "b", 0.9)], auc=0.9, auc_note=None)
    )
    assert "FAIL" in fail_text
    assert "PASS" in pass_text
    assert "0.900" in pass_text


def test_cmd_template_writes_file_and_refuses_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import argparse

    out = tmp_path / "people.yaml"
    cmd_template(argparse.Namespace(out=str(out), force=False))
    assert out.exists()
    rc = cmd_template(argparse.Namespace(out=str(out), force=False))
    assert rc == 1
    rc2 = cmd_template(argparse.Namespace(out=str(out), force=True))
    assert rc2 == 0


def test_scenario_list_defaults_to_cold_open() -> None:
    from regulars.ops.persona_diagnostic import _scenario_list

    assert _scenario_list(None) == [COLD_OPEN]
    assert _scenario_list("cold_open,dead_air") == [COLD_OPEN, DEAD_AIR]


def test_cmd_personas_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse

    people = tmp_path / "people.yaml"
    people.write_text(
        "people:\n"
        '  - id: priya\n    answers: ["a", "b", "c"]\n'
        '  - id: sam\n    answers: ["x", "y", "z"]\n'
    )
    out = tmp_path / "personas"
    fake = FakeLLM(replies=[persona_json("priya"), persona_json("sam")])
    monkeypatch.setattr("regulars.ops.persona_diagnostic.llm_from_env", lambda role: fake)
    rc = cmd_personas(argparse.Namespace(input=str(people), out=str(out), llm="scoring"))
    assert rc == 0
    assert (out / "priya.json").exists()
    assert (out / "sam.json").exists()


def test_cmd_variance_end_to_end_with_yes_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import argparse

    from regulars.core import PersonaSpec

    personas_dir = tmp_path / "personas"
    personas_dir.mkdir()
    write_personas(
        {
            "priya": PersonaSpec.model_validate_json(persona_json("priya")),
            "sam": PersonaSpec.model_validate_json(persona_json("sam")),
        },
        personas_dir,
    )
    fake = FakeLLM(responder=lambda system, messages: "sure, sounds good, let's do it")
    monkeypatch.setattr("regulars.ops.persona_diagnostic.llm_from_env", lambda role: fake)
    rc = cmd_variance(
        argparse.Namespace(
            personas=str(personas_dir),
            input=None,
            llm="enactment",
            turns=4,
            seeds=1,
            scenarios="cold_open",
            full=False,
            out=str(tmp_path / "report.md"),
            yes=True,
        )
    )
    assert rc in (0, 1)
    assert (tmp_path / "report.md").exists()
