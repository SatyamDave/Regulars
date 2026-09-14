from __future__ import annotations

import json

from regulars.core.llm import FakeLLM
from regulars.core.transcript import Transcript, Turn
from regulars.features.questions import (
    HeuristicQuestionClassifier,
    LLMQuestionClassifier,
    QuestionClassifier,
    follow_up_rate,
    follow_up_rate_from_labels,
    gini,
    is_question,
    question_counts,
    question_gini,
)
from tests.features.conftest import A, B, dyad

clf = HeuristicQuestionClassifier()


def test_is_question_surface_forms() -> None:
    assert is_question("Where did you grow up?")
    assert is_question("do you climb")  # aux inversion, no mark
    assert is_question("why though")
    assert not is_question("I climb on Saturdays.")
    assert not is_question("")


def test_heuristic_labels_on_flow(flow: Transcript) -> None:
    labels = clf.classify_turns(flow.spoken())
    assert labels[0] == "none"
    assert labels[1] == "follow_up"  # "which gym?" echoes "gym"
    assert labels[5] == "follow_up"  # opens with "so"
    assert labels[6] == "mirror"  # "what about you" right after being asked
    assert labels[15] == "follow_up"  # opens with "wait"


def test_heuristic_labels_on_interview(interview: Transcript) -> None:
    labels = clf.classify_turns(interview.spoken())
    assert labels[0] == "new_topic"
    assert labels[4] == "follow_up"  # shares "accounting" with the answer
    assert all(lab == "none" for i, lab in enumerate(labels) if i % 2 == 1)


def test_rhetorical_and_tag_questions() -> None:
    t = dyad([(A, "Traffic was awful."), (B, "Who knows why they closed the bridge?")])
    assert clf.classify_turns(t.spoken())[1] == "rhetorical"
    t = dyad([(A, "It rained all week."), (B, "Typical for March, right?")])
    assert clf.classify_turns(t.spoken())[1] == "rhetorical"


def test_mirror_by_overlap() -> None:
    t = dyad([(A, "What did you cook this weekend?"), (B, "What did you cook?")])
    assert clf.classify_turns(t.spoken())[1] == "mirror"


def test_follow_up_balance_flow_beats_interview(flow: Transcript, interview: Transcript) -> None:
    fl = clf.classify_turns(flow.spoken())
    il = clf.classify_turns(interview.spoken())
    flow_ab = follow_up_rate_from_labels(flow.spoken(), fl, A)
    flow_ba = follow_up_rate_from_labels(flow.spoken(), fl, B)
    int_ab = follow_up_rate_from_labels(interview.spoken(), il, A)
    int_ba = follow_up_rate_from_labels(interview.spoken(), il, B)
    assert min(flow_ab, flow_ba) > min(int_ab, int_ba)
    assert flow_ab + flow_ba > int_ab + int_ba
    assert int_ba == 0.0  # the interviewee never asks


async def test_follow_up_rate_is_directed(flow: Transcript) -> None:
    ab = await follow_up_rate(flow, A, B, clf)
    ba = await follow_up_rate(flow, B, A, clf)
    assert ab != ba
    assert 0.0 < ab < ba <= 1.0


async def test_follow_up_rate_restricts_group_chat() -> None:
    group = Transcript(
        participants=[A, B, "cy"],
        turns=[
            Turn(speaker_id=A, text="I started climbing at the gym."),
            Turn(speaker_id="cy", text="Boring."),
            Turn(speaker_id=B, text="Which gym?"),
        ],
        source="sim",
    )
    assert await follow_up_rate(group, B, A, clf) == 1.0


def test_gini_formula() -> None:
    assert gini([5, 5]) == 0.0
    assert gini([10, 0]) == 1.0
    assert gini([0, 0]) == 0.0
    assert 0.0 < gini([4, 6]) < 0.5


def test_question_gini_interview_beats_flow(
    flow: Transcript, interview: Transcript, dead: Transcript
) -> None:
    assert question_gini(interview) > question_gini(flow)
    assert question_gini(interview) == 1.0
    assert question_gini(dead) == 1.0  # only ana asked anything
    counts = question_counts(interview)
    assert counts[A] == 10 and counts[B] == 0


def test_question_counts_exclude_rhetorical() -> None:
    t = dyad([(A, "Who knows why they closed the bridge?"), (B, "Which bridge?")])
    assert question_counts(t) == {A: 0, B: 1}


async def test_llm_classifier_one_call_and_json(flow: Transcript) -> None:
    n = len(flow.spoken())
    scripted = ["follow_up"] * n
    fake = FakeLLM(replies=[json.dumps(scripted)])
    llm_clf: QuestionClassifier = LLMQuestionClassifier(fake)
    labels = await llm_clf.classify(flow)
    assert labels == scripted
    assert len(fake.calls) == 1
    system, messages = fake.calls[0]
    assert "question type" in system
    assert f"{n}. [{B}]" in messages[0].content  # every turn numbered in the one prompt


async def test_llm_classifier_accepts_fenced_object(flow: Transcript) -> None:
    n = len(flow.spoken())
    body = json.dumps({"labels": ["New Topic"] * n})
    fake = FakeLLM(replies=[f"```json\n{body}\n```"])
    labels = await LLMQuestionClassifier(fake).classify(flow)
    assert labels == ["new_topic"] * n


async def test_llm_classifier_falls_back_per_entry(flow: Transcript) -> None:
    heuristic = clf.classify_turns(flow.spoken())
    scripted: list[str | int] = ["rhetorical"] + ["bogus"] * (len(heuristic) - 2) + [7]
    fake = FakeLLM(replies=[json.dumps(scripted)])
    labels = await LLMQuestionClassifier(fake).classify(flow)
    assert labels[0] == "rhetorical"
    assert labels[1:] == heuristic[1:]


async def test_llm_classifier_falls_back_wholesale_on_bad_response(flow: Transcript) -> None:
    fake = FakeLLM(replies=["I cannot help with that."])
    labels = await LLMQuestionClassifier(fake).classify(flow)
    assert labels == clf.classify_turns(flow.spoken())
    fake = FakeLLM(replies=[json.dumps(["none", "none"])])  # wrong length
    labels = await LLMQuestionClassifier(fake).classify(flow)
    assert labels == clf.classify_turns(flow.spoken())


async def test_llm_classifier_empty_transcript_makes_no_call() -> None:
    fake = FakeLLM()
    empty = Transcript(participants=[A, B], turns=[], source="sim")
    assert await LLMQuestionClassifier(fake).classify(empty) == []
    assert fake.calls == []
