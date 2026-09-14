from __future__ import annotations

import json

import pytest

from regulars.core.llm import FakeLLM
from regulars.core.transcript import Transcript
from regulars.features.disclosure import (
    DisclosureScorer,
    HeuristicDisclosureScorer,
    LLMDisclosureScorer,
    asymmetry_from_depths,
    disclosure_asymmetry,
    disclosure_reciprocity,
    lag1_pairs,
    reciprocity_from_depths,
)
from tests.features.conftest import A, B, dyad

scorer = HeuristicDisclosureScorer()


def test_depth_ladder() -> None:
    assert scorer.score_text("Nice weather today.") == 0
    assert scorer.score_text("ok") == 0
    assert scorer.score_text("I moved here in March for the job.") == 1
    assert scorer.score_text("I was nervous about moving here.") == 2
    assert scorer.score_text("Honestly I was terrified about moving here.") == 3
    long = "I was scared about moving here because I did not know anyone in the whole city"
    assert scorer.score_text(long) == 3
    assert scorer.score_text("To be honest I don't talk about my dad much.") == 3


def test_lag1_pairs_only_adjacent_a_then_b() -> None:
    t = dyad([(A, "x"), (B, "y"), (B, "z"), (A, "w"), (A, "v"), (B, "u")])
    depths = [1, 2, 3, 0, 1, 3]
    assert lag1_pairs(t.spoken(), depths, A, B) == [(1, 2), (1, 3)]
    assert lag1_pairs(t.spoken(), depths, B, A) == [(3, 0)]


def test_reciprocity_needs_four_pairs_and_variance() -> None:
    t = dyad([(A, "a"), (B, "b")] * 3)
    assert reciprocity_from_depths(t.spoken(), [3, 3, 0, 0, 2, 2], A, B) == 0.0  # 3 pairs
    t = dyad([(A, "a"), (B, "b")] * 4)
    assert reciprocity_from_depths(t.spoken(), [1] * 8, A, B) == 0.0  # no variance


def test_reciprocity_perfect_mirroring() -> None:
    t = dyad([(A, "a"), (B, "b")] * 5)
    depths = [0, 0, 3, 3, 1, 1, 2, 2, 3, 3]
    assert reciprocity_from_depths(t.spoken(), depths, A, B) == pytest.approx(1.0)
    inverted = [0, 3, 3, 0, 1, 2, 2, 1, 3, 0]
    assert reciprocity_from_depths(t.spoken(), inverted, A, B) < 0


async def test_reciprocity_and_asymmetry_via_scripted_llm_scorer() -> None:
    t = dyad([(A, "a"), (B, "b")] * 5)
    depths = [3, 3, 0, 0, 2, 2, 1, 1, 3, 3]
    llm: DisclosureScorer = LLMDisclosureScorer(FakeLLM(replies=[json.dumps(depths)]))
    assert await disclosure_reciprocity(t, llm, A, B) == pytest.approx(1.0)
    carried = [3, 0, 3, 0, 2, 0, 3, 0, 3, 0]
    llm = LLMDisclosureScorer(FakeLLM(replies=[json.dumps(carried)]))
    assert await disclosure_asymmetry(t, llm) == pytest.approx(2.8)
    assert asymmetry_from_depths(t.spoken(), carried, A, B) == pytest.approx(2.8)


def test_fixture_ordering(flow: Transcript, interview: Transcript) -> None:
    fd = scorer.score_turns(flow.spoken())
    idd = scorer.score_turns(interview.spoken())
    assert sum(fd) > sum(idd)
    assert reciprocity_from_depths(flow.spoken(), fd, A, B) > reciprocity_from_depths(
        interview.spoken(), idd, A, B
    )
    assert reciprocity_from_depths(flow.spoken(), fd, A, B) > 0
    assert asymmetry_from_depths(interview.spoken(), idd, A, B) == 0.0
    assert asymmetry_from_depths(flow.spoken(), fd, A, B) < 1.0  # both carry it


async def test_llm_scorer_one_call_and_fallback(flow: Transcript) -> None:
    n = len(flow.spoken())
    heuristic = scorer.score_turns(flow.spoken())
    raw: list[object] = ["2"] * (n - 2) + [9, "deep"]
    fake = FakeLLM(replies=[json.dumps(raw)])
    out = await LLMDisclosureScorer(fake).score(flow)
    assert out[: n - 2] == [2] * (n - 2)
    assert out[n - 2 :] == heuristic[n - 2 :]
    assert len(fake.calls) == 1
    assert "self-disclosure" in fake.calls[0][0]
    fake = FakeLLM(replies=[json.dumps({"labels": [1.0] * n})])
    assert await LLMDisclosureScorer(fake).score(flow) == [1] * n
    assert B in flow.participants
