"""Real-chat-only features (§4 "Real chat only"): latency, reply graph, tapbacks.

Every function here degrades to "absent" (empty list / empty dict) when the transcript is
simulated or lacks timestamps. Emitted keys are exactly ``REAL_ONLY_FEATURES``.

Latency follows Templeton et al. (PNAS 2022) with the §4 normalization::

    latency_norm = log(raw_ms) - median(log latency of this user, this hour-of-day)

Gaps over four hours are dropped (sleep or work, not disengagement). The features use the
within-conversation distribution: connected moments are *faster than the user's own baseline*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from statistics import median

import networkx as nx
from pydantic import BaseModel, Field

from regulars.core.features import (
    LATENCY_NORM_FAST_SHARE,
    LATENCY_NORM_MEDIAN,
    REAL_ONLY_FEATURES,
    REPLY_WEIGHT,
    TAPBACK_RATE,
)
from regulars.core.transcript import Transcript, Turn

MAX_GAP_MS = 4 * 60 * 60 * 1000
"""Gaps longer than this are not disengagement; they are dropped before any statistic."""

MIN_HOUR_SAMPLES = 5
"""An hour-of-day bucket with fewer samples than this falls back to the user's overall median."""


@dataclass(frozen=True, slots=True)
class LatencySample:
    """One reply gap: ``responder`` answered ``to`` after ``raw_ms``."""

    raw_ms: int
    hour_of_day: int
    turn_index: int

    @property
    def log_ms(self) -> float:
        return math.log(max(self.raw_ms, 1))


class UserBaseline(BaseModel):
    """Per-user, per-hour-of-day median of ``log(latency_ms)``.

    ``hourly_log_median`` holds only hours with at least ``MIN_HOUR_SAMPLES`` samples;
    ``overall_log_median`` is the fallback. Fit with :func:`fit_baseline` over the user's whole
    message history, not the conversation being scored.
    """

    user_id: str
    hourly_log_median: dict[int, float] = Field(default_factory=dict)
    overall_log_median: float | None = None
    sample_count: int = 0

    def expected_log(self, hour_of_day: int) -> float | None:
        hourly = self.hourly_log_median.get(hour_of_day)
        return hourly if hourly is not None else self.overall_log_median


def hour_of_day(sent_at_ms: int, tz: tzinfo = UTC) -> int:
    return datetime.fromtimestamp(sent_at_ms / 1000, tz=tz).hour


def response_latencies(
    transcript: Transcript, responder: str, to: str, *, tz: tzinfo = UTC
) -> list[LatencySample]:
    """Gaps for each spoken turn by ``responder`` that answers ``to``.

    A turn answers ``to`` when it explicitly ``reply_to``s a spoken turn by ``to``, or, lacking an
    explicit reply, when the immediately preceding spoken turn is by ``to``. Tapbacks are neither
    turns nor targets. Gaps over :data:`MAX_GAP_MS` or negative gaps are dropped. Returns ``[]``
    when the transcript lacks timing.
    """
    if not transcript.has_timing():
        return []
    turns = transcript.turns
    samples: list[LatencySample] = []
    prev_spoken: Turn | None = None
    for index, turn in enumerate(turns):
        if turn.is_tapback:
            continue
        target = _answered_turn(turns, turn, prev_spoken)
        prev_spoken = turn
        if turn.speaker_id != responder or target is None or target.speaker_id != to:
            continue
        if turn.sent_at_ms is None or target.sent_at_ms is None:
            continue
        gap = turn.sent_at_ms - target.sent_at_ms
        if gap < 0 or gap > MAX_GAP_MS:
            continue
        samples.append(LatencySample(gap, hour_of_day(turn.sent_at_ms, tz), index))
    return samples


def _answered_turn(turns: list[Turn], turn: Turn, prev_spoken: Turn | None) -> Turn | None:
    if turn.reply_to is not None:
        target = turns[turn.reply_to]
        return None if target.is_tapback else target
    return prev_spoken


def fit_user_baseline(user_id: str, samples: list[LatencySample]) -> UserBaseline:
    by_hour: dict[int, list[float]] = {}
    for sample in samples:
        by_hour.setdefault(sample.hour_of_day, []).append(sample.log_ms)
    hourly = {hour: median(logs) for hour, logs in by_hour.items() if len(logs) >= MIN_HOUR_SAMPLES}
    overall = median(s.log_ms for s in samples) if samples else None
    return UserBaseline(
        user_id=user_id,
        hourly_log_median=hourly,
        overall_log_median=overall,
        sample_count=len(samples),
    )


def fit_baseline(samples_by_user: dict[str, list[LatencySample]]) -> dict[str, UserBaseline]:
    """Fit one :class:`UserBaseline` per user from all of that user's reply gaps."""
    return {user: fit_user_baseline(user, samples) for user, samples in samples_by_user.items()}


def normalize(samples: list[LatencySample], baseline: UserBaseline) -> list[float]:
    """``log(raw_ms) - baseline`` per sample. Empty when the baseline has no data at all."""
    out: list[float] = []
    for sample in samples:
        expected = baseline.expected_log(sample.hour_of_day)
        if expected is None:
            return []
        out.append(sample.log_ms - expected)
    return out


def latency_features(
    transcript: Transcript, a: str, b: str, baseline_a: UserBaseline, *, tz: tzinfo = UTC
) -> dict[str, float]:
    """Directed latency features for ``a`` replying to ``b``; ``{}`` without timing or samples.

    ``latency_norm_median`` below zero means ``a`` answers ``b`` faster than ``a``'s own habit;
    ``latency_norm_fast_share`` is the share of such faster-than-baseline replies.
    """
    normalized = normalize(response_latencies(transcript, a, b, tz=tz), baseline_a)
    if not normalized:
        return {}
    return {
        LATENCY_NORM_MEDIAN: float(median(normalized)),
        LATENCY_NORM_FAST_SHARE: sum(1 for x in normalized if x < 0) / len(normalized),
    }


def reply_graph(transcript: Transcript) -> nx.DiGraph:
    """Directed graph weighted a→b by the number of a's spoken turns answering b.

    Explicit ``reply_to`` wins; otherwise the answered turn is the most recent spoken turn by a
    different speaker. Replies to oneself add no edge. Every participant is a node.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(transcript.participants)
    turns = transcript.turns
    recent_spoken: list[Turn] = []
    for turn in turns:
        if turn.is_tapback:
            continue
        if turn.reply_to is not None:
            target = turns[turn.reply_to]
            answered = None if target.is_tapback else target
        else:
            answered = next(
                (t for t in reversed(recent_spoken) if t.speaker_id != turn.speaker_id), None
            )
        recent_spoken.append(turn)
        if answered is None or answered.speaker_id == turn.speaker_id:
            continue
        a, b = turn.speaker_id, answered.speaker_id
        graph.add_edge(a, b, weight=graph.get_edge_data(a, b, {"weight": 0})["weight"] + 1)
    return graph


def reply_weight(graph: nx.DiGraph, a: str, b: str) -> float:
    """Weight of a→b as a share of a's total out-weight; ``0.0`` if a answered nobody."""
    total = sum(float(data["weight"]) for _, _, data in graph.out_edges(a, data=True))
    if total == 0:
        return 0.0
    edge = graph.get_edge_data(a, b)
    return float(edge["weight"]) / total if edge else 0.0


def tapback_rate(transcript: Transcript, a: str, b: str) -> float:
    """Tapbacks by ``a`` on ``b``'s turns, per spoken turn of ``b``; ``0.0`` if b never spoke."""
    spoken_by_b = len(transcript.turns_by(b))
    if spoken_by_b == 0:
        return 0.0
    turns = transcript.turns
    count = sum(
        1
        for t in turns
        if t.is_tapback
        and t.speaker_id == a
        and t.tapback_target is not None
        and turns[t.tapback_target].speaker_id == b
    )
    return count / spoken_by_b


def extract_real_features(
    transcript: Transcript,
    a: str,
    b: str,
    baselines: dict[str, UserBaseline],
    *,
    tz: tzinfo = UTC,
) -> tuple[dict[str, float], dict[str, float]]:
    """Real-only features for ``a→b`` and ``b→a``. Both ``{}`` for sim or untimed transcripts.

    Latency keys appear only when the replier has a fitted baseline and at least one reply gap;
    reply weight and tapback rate are always present for real, timed transcripts.
    """
    if transcript.source != "real" or not transcript.has_timing():
        return {}, {}
    graph = reply_graph(transcript)

    def one_way(x: str, y: str) -> dict[str, float]:
        features: dict[str, float] = {}
        baseline = baselines.get(x)
        if baseline is not None:
            features.update(latency_features(transcript, x, y, baseline, tz=tz))
        features[REPLY_WEIGHT] = reply_weight(graph, x, y)
        features[TAPBACK_RATE] = tapback_rate(transcript, x, y)
        if not set(features) <= set(REAL_ONLY_FEATURES):
            raise RuntimeError(f"real extractor emitted non-real keys: {sorted(features)}")
        return features

    return one_way(a, b), one_way(b, a)
