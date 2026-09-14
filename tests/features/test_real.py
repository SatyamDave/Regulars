import math

import pytest

from regulars.core.features import (
    LATENCY_NORM_FAST_SHARE,
    LATENCY_NORM_MEDIAN,
    REAL_ONLY_FEATURES,
    REPLY_WEIGHT,
    TAPBACK_RATE,
)
from regulars.core.transcript import Transcript, Turn
from regulars.features.real import (
    MAX_GAP_MS,
    LatencySample,
    UserBaseline,
    extract_real_features,
    fit_baseline,
    latency_features,
    normalize,
    reply_graph,
    reply_weight,
    response_latencies,
    tapback_rate,
)

T0 = 10 * 60 * 60 * 1000  # 10:00 UTC, 1970-01-01
MIN = 60_000


def real(*turns: Turn, participants: list[str] | None = None) -> Transcript:
    return Transcript(
        participants=participants or ["a", "b", "c"], turns=list(turns), source="real"
    )


def test_latency_from_timestamps_and_implicit_previous_turn() -> None:
    t = real(
        Turn(speaker_id="b", text="hi", sent_at_ms=T0),
        Turn(speaker_id="a", text="hey", sent_at_ms=T0 + 2 * MIN),
        Turn(speaker_id="a", text="how are you", sent_at_ms=T0 + 3 * MIN),  # follows self
        Turn(speaker_id="b", text="good", sent_at_ms=T0 + 4 * MIN),
    )
    samples = response_latencies(t, "a", "b")
    assert samples == [LatencySample(raw_ms=2 * MIN, hour_of_day=10, turn_index=1)]
    assert response_latencies(t, "b", "a") == [LatencySample(MIN, 10, 3)]


def test_latency_explicit_reply_to_overrides_previous_turn() -> None:
    t = real(
        Turn(speaker_id="b", text="q?", sent_at_ms=T0),
        Turn(speaker_id="c", text="unrelated", sent_at_ms=T0 + MIN),
        Turn(speaker_id="a", text="answer", sent_at_ms=T0 + 5 * MIN, reply_to=0),
    )
    assert response_latencies(t, "a", "b") == [LatencySample(5 * MIN, 10, 2)]
    assert response_latencies(t, "a", "c") == []  # explicit reply is not an answer to c


def test_gaps_over_four_hours_dropped_and_tapbacks_ignored() -> None:
    t = real(
        Turn(speaker_id="b", text="night", sent_at_ms=T0),
        Turn(speaker_id="a", text="morning", sent_at_ms=T0 + MAX_GAP_MS + 1),
        Turn(speaker_id="b", text="👍", sent_at_ms=T0 + MAX_GAP_MS + 2, tapback_target=1),
        Turn(speaker_id="b", text="hi", sent_at_ms=T0 + MAX_GAP_MS + 3 * MIN),
        Turn(speaker_id="a", text="yo", sent_at_ms=T0 + MAX_GAP_MS + 3 * MIN + 30_000),
    )
    samples = response_latencies(t, "a", "b")
    assert [s.raw_ms for s in samples] == [30_000]
    # b's tapback is not a turn: b→a gap is measured from a's message to b's spoken "hi".
    assert [s.raw_ms for s in response_latencies(t, "b", "a")] == [3 * MIN - 1]


def test_latency_without_timing_is_empty() -> None:
    t = real(Turn(speaker_id="b", text="hi"), Turn(speaker_id="a", text="hey"))
    assert response_latencies(t, "a", "b") == []


def test_fit_baseline_hourly_with_fallback() -> None:
    hour_10 = [LatencySample(2**k * 1000, 10, k) for k in range(5)]  # log medians: k=2
    hour_11 = [LatencySample(60_000, 11, 9), LatencySample(120_000, 11, 10)]  # <5: fallback
    baselines = fit_baseline({"a": hour_10 + hour_11, "z": []})
    a = baselines["a"]
    assert a.sample_count == 7
    assert a.hourly_log_median == {10: pytest.approx(math.log(4000))}
    logs = sorted(math.log(s.raw_ms) for s in hour_10 + hour_11)
    assert a.overall_log_median == pytest.approx(logs[3])
    assert a.expected_log(10) == pytest.approx(math.log(4000))
    assert a.expected_log(11) == pytest.approx(logs[3])
    assert baselines["z"].expected_log(10) is None


def test_normalize_hand_computed() -> None:
    baseline = UserBaseline(
        user_id="a", hourly_log_median={10: math.log(60_000)}, overall_log_median=math.log(30_000)
    )
    samples = [
        LatencySample(30_000, 10, 1),
        LatencySample(120_000, 10, 2),
        LatencySample(60_000, 3, 3),
    ]
    got = normalize(samples, baseline)
    assert got == pytest.approx([math.log(0.5), math.log(2.0), math.log(2.0)])
    assert normalize(samples, UserBaseline(user_id="a")) == []


def test_latency_features_median_and_fast_share() -> None:
    baseline = UserBaseline(user_id="a", overall_log_median=math.log(60_000))
    t = real(
        Turn(speaker_id="b", text="1", sent_at_ms=T0),
        Turn(speaker_id="a", text="fast", sent_at_ms=T0 + 15_000),  # 15s  -> log(0.25) < 0
        Turn(speaker_id="b", text="2", sent_at_ms=T0 + MIN),
        Turn(speaker_id="a", text="slow", sent_at_ms=T0 + MIN + 240_000),  # 4m -> log(4) > 0
        Turn(speaker_id="b", text="3", sent_at_ms=T0 + 10 * MIN),
        Turn(speaker_id="a", text="fast", sent_at_ms=T0 + 10 * MIN + 30_000),  # 30s -> log(.5)
    )
    feats = latency_features(t, "a", "b", baseline)
    assert feats[LATENCY_NORM_MEDIAN] == pytest.approx(math.log(0.5))
    assert feats[LATENCY_NORM_FAST_SHARE] == pytest.approx(2 / 3)
    assert latency_features(t, "b", "c", baseline) == {}


def test_reply_graph_explicit_and_implicit() -> None:
    t = real(
        Turn(speaker_id="a", text="1", sent_at_ms=T0),
        Turn(speaker_id="b", text="2", sent_at_ms=T0 + 1),  # implicit b->a
        Turn(speaker_id="b", text="3", sent_at_ms=T0 + 2),  # follows self -> still b->a
        Turn(speaker_id="c", text="4", sent_at_ms=T0 + 3, reply_to=0),  # explicit c->a
        Turn(speaker_id="a", text="5", sent_at_ms=T0 + 4),  # implicit a->c
        Turn(speaker_id="a", text="6", sent_at_ms=T0 + 5, reply_to=2),  # explicit a->b
        Turn(speaker_id="c", text="❤️", sent_at_ms=T0 + 6, tapback_target=5),  # not a reply
        Turn(speaker_id="a", text="7", sent_at_ms=T0 + 7, reply_to=4),  # self reply, no edge
    )
    g = reply_graph(t)
    assert set(g.nodes) == {"a", "b", "c"}
    assert g["b"]["a"]["weight"] == 2
    assert g["c"]["a"]["weight"] == 1
    assert g["a"]["c"]["weight"] == 1
    assert g["a"]["b"]["weight"] == 1
    assert reply_weight(g, "a", "b") == pytest.approx(0.5)
    assert reply_weight(g, "b", "a") == pytest.approx(1.0)
    assert reply_weight(g, "c", "b") == 0.0
    assert reply_weight(g, "a", "a") == 0.0


def test_reply_weight_no_out_edges() -> None:
    g = reply_graph(real(Turn(speaker_id="a", text="alone", sent_at_ms=T0)))
    assert reply_weight(g, "a", "b") == 0.0


def test_tapback_rate_is_directed() -> None:
    t = real(
        Turn(speaker_id="b", text="1", sent_at_ms=T0),
        Turn(speaker_id="b", text="2", sent_at_ms=T0 + 1),
        Turn(speaker_id="a", text="👍", sent_at_ms=T0 + 2, tapback_target=0),
        Turn(speaker_id="a", text="❤️", sent_at_ms=T0 + 3, tapback_target=1),
        Turn(speaker_id="a", text="3", sent_at_ms=T0 + 4),
        Turn(speaker_id="c", text="😂", sent_at_ms=T0 + 5, tapback_target=4),
    )
    assert tapback_rate(t, "a", "b") == pytest.approx(1.0)  # 2 tapbacks / 2 b messages
    assert tapback_rate(t, "b", "a") == 0.0
    assert tapback_rate(t, "c", "a") == pytest.approx(1.0)
    assert tapback_rate(t, "a", "c") == 0.0  # c never spoke


def test_extract_real_features_both_directions() -> None:
    t = real(
        Turn(speaker_id="b", text="1", sent_at_ms=T0),
        Turn(speaker_id="a", text="2", sent_at_ms=T0 + 30_000),
        Turn(speaker_id="b", text="👍", sent_at_ms=T0 + 31_000, tapback_target=1),
        Turn(speaker_id="b", text="3", sent_at_ms=T0 + 2 * MIN),
    )
    baselines = {"a": UserBaseline(user_id="a", overall_log_median=math.log(60_000))}
    ab, ba = extract_real_features(t, "a", "b", baselines)
    assert set(ab) <= set(REAL_ONLY_FEATURES) and set(ba) <= set(REAL_ONLY_FEATURES)
    assert ab[LATENCY_NORM_MEDIAN] == pytest.approx(math.log(0.5))
    assert ab[LATENCY_NORM_FAST_SHARE] == 1.0
    assert ab[REPLY_WEIGHT] == 1.0
    assert ab[TAPBACK_RATE] == 0.0
    assert LATENCY_NORM_MEDIAN not in ba  # b has no baseline
    assert ba[REPLY_WEIGHT] == 1.0
    assert ba[TAPBACK_RATE] == 1.0


def test_extract_real_features_sim_or_untimed_is_empty() -> None:
    sim = Transcript(
        participants=["a", "b"],
        turns=[Turn(speaker_id="a", text="hi"), Turn(speaker_id="b", text="hey")],
        source="sim",
    )
    assert extract_real_features(sim, "a", "b", {}) == ({}, {})
    untimed = Transcript(
        participants=["a", "b"],
        turns=[Turn(speaker_id="a", text="hi", sent_at_ms=T0), Turn(speaker_id="b", text="hey")],
        source="real",
    )
    assert extract_real_features(untimed, "a", "b", {}) == ({}, {})
