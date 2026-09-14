from __future__ import annotations

import json

from regulars.core.llm import FakeLLM
from regulars.core.transcript import Transcript
from regulars.features.topics import (
    NO_TOPIC,
    HeuristicTopicSegmenter,
    LLMTopicSegmenter,
    TopicSegmenter,
    normalize_segments,
    segment_lengths,
    topic_survival,
    topic_turns_p90,
)
from tests.features.conftest import A, B, dyad

seg = HeuristicTopicSegmenter()


def test_segment_lengths_and_no_topic() -> None:
    assert segment_lengths([0, 0, 0, 1, -1, -1, 2, 2]) == [3, 1, 2]
    assert segment_lengths([]) == []
    assert segment_lengths([NO_TOPIC, NO_TOPIC]) == []


def test_topic_survival_and_p90_formulas() -> None:
    segs = [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2]  # lengths 4, 2, 5
    assert topic_survival(segs) == 2 / 3
    assert topic_turns_p90(segs) == 4.8  # linear interpolation of [2, 4, 5]
    assert topic_survival([]) == 0.0
    assert topic_turns_p90([]) == 0.0


def test_normalize_segments() -> None:
    assert normalize_segments([0, 0, 2, 2, 5, -1, 7, 7]) == [0, 0, 1, 1, 2, -1, 3, 3]
    assert normalize_segments([3, 3, 0, 0]) == [0, 0, 1, 1]
    assert normalize_segments([]) == []


def test_heuristic_topic_ordering(
    flow: Transcript, interview: Transcript, dead: Transcript
) -> None:
    fs, is_, ds = (seg.segment_turns(t.spoken()) for t in (flow, interview, dead))
    assert topic_survival(fs) > topic_survival(is_)
    assert topic_survival(fs) > topic_survival(ds)
    assert topic_survival(ds) == 0.0
    assert topic_turns_p90(fs) > topic_turns_p90(is_)
    assert topic_turns_p90(fs) > topic_turns_p90(ds)
    assert max(fs) >= 1  # flow does move between topics; it just dwells on them


def test_heuristic_new_topic_question_starts_segment(interview: Transcript) -> None:
    segs = seg.segment_turns(interview.spoken())
    assert segs[0] == 0
    assert segs[2] == 1  # "What do you do for work?" opens a new topic
    assert segs[4] == segs[3]  # the one follow-up stays on the accounting topic


def test_heuristic_contentless_run_kills_topic() -> None:
    t = dyad(
        [
            (A, "I started climbing at the bouldering gym last month."),
            (B, "cool"),
            (A, "yeah"),
            (B, "ok"),
            (A, "Anyway my sister is visiting from Denver next week."),
        ]
    )
    segs = seg.segment_turns(t.spoken())
    assert segs[0] == 0
    assert segs[2] == 0  # second contentless turn is the last one on the topic
    assert segs[3] == NO_TOPIC
    assert segs[4] == 1


def test_heuristic_low_jaccard_starts_new_topic() -> None:
    t = dyad(
        [
            (A, "I started climbing at the bouldering gym last month."),
            (B, "The climbing gym near the river is great for bouldering."),
            (A, "My landlord raised the rent again and the heating is broken."),
        ]
    )
    segs = seg.segment_turns(t.spoken())
    assert segs == [0, 0, 1]


async def test_llm_segmenter_one_call(flow: Transcript) -> None:
    n = len(flow.spoken())
    ids = [0] * (n // 2) + [1] * (n - n // 2)
    fake = FakeLLM(replies=[json.dumps(ids)])
    segmenter: TopicSegmenter = LLMTopicSegmenter(fake)
    assert await segmenter.segment(flow) == ids
    assert len(fake.calls) == 1
    assert "segment" in fake.calls[0][0]


async def test_llm_segmenter_normalizes_and_falls_back(flow: Transcript) -> None:
    n = len(flow.spoken())
    heuristic = seg.segment_turns(flow.spoken())
    raw = ["4"] * (n - 1) + ["x"]  # numeric strings accepted; "x" falls back to heuristic
    fake = FakeLLM(replies=[json.dumps(raw)])
    out = await LLMTopicSegmenter(fake).segment(flow)
    assert len(out) == n
    assert out[: n - 1] == [0] * (n - 1)  # one run of "4" renumbered to topic 0
    assert out[-1] in (0, 1, NO_TOPIC)  # heuristic's last id, renumbered after the run
    fake = FakeLLM(replies=["nope"])
    assert await LLMTopicSegmenter(fake).segment(flow) == normalize_segments(heuristic)


def test_dead_air_turns_before_first_topic(dead: Transcript) -> None:
    segs = seg.segment_turns(dead.spoken())
    assert segs[0] == NO_TOPIC and segs[1] == NO_TOPIC
    assert segs[2] == 0
    assert B in dead.participants  # fixture sanity
