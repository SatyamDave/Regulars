"""Flat moments and repair (docs/technical-architecture.md §4).

A flat moment is a spoken turn that is one word, a deflection ("anyway", "sure", "ok", "cool"),
or the death of a topic (a segment that ended within ``short_topic_turns`` turns). A repair is
the next turn, by either speaker and within ``window`` turns, that asks a question or brings
>= ``content_words`` words of content. The ``dead_air`` scenario exists to elicit who repairs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from regulars.core.transcript import Transcript, Turn
from regulars.features._text import tokenize, word_count, words
from regulars.features.questions import ASKING, QuestionType, is_question
from regulars.features.topics import NO_TOPIC, HeuristicTopicSegmenter

FlatKind = Literal["one_word", "deflection", "topic_death"]

DEFLECTIONS: frozenset[str] = words(
    """
    anyway anyways sure ok okay k kk cool fine whatever nice yeah yep yup nope lol haha right mhm
    hm hmm meh idk true same word gotcha alright aight ya cool_cool i_guess if_you_say_so
    moving_on fair_enough good_to_know makes_sense
    """
)
_DEFLECTION_OPENERS = ("anyway", "anyways", "moving on", "whatever")


@dataclass(frozen=True)
class FlatMoment:
    index: int  # position in transcript.spoken()
    kind: FlatKind
    speaker_id: str
    repaired_at: int | None = None
    repaired_by: str | None = None

    @property
    def repaired(self) -> bool:
        return self.repaired_at is not None

    @property
    def latency_turns(self) -> int | None:
        return None if self.repaired_at is None else self.repaired_at - self.index


def is_deflection(text: str) -> bool:
    tokens = tokenize(text)
    if not tokens:
        return False
    if "_".join(tokens) in DEFLECTIONS or all(tok in DEFLECTIONS for tok in tokens):
        return True
    lowered = text.strip().lower()
    return lowered.startswith(_DEFLECTION_OPENERS) and len(tokens) <= 4


def flat_kind(turn: Turn) -> FlatKind | None:
    """One-word / deflection test on a single turn (topic death needs the segmentation)."""
    if is_question(turn.text):
        return None
    n = word_count(turn.text)
    if n <= 1:
        return "one_word"
    if is_deflection(turn.text):
        return "deflection"
    return None


def detect_flat_moments(
    turns: Sequence[Turn],
    segments: Sequence[int] | None = None,
    short_topic_turns: int = 2,
) -> list[FlatMoment]:
    """Flat moments in spoken-turn order; a turn is reported once (surface kind wins)."""
    if segments is None:
        segments = HeuristicTopicSegmenter().segment_turns(turns)
    flats: list[FlatMoment] = []
    seen: set[int] = set()
    for i, t in enumerate(turns):
        kind = flat_kind(t)
        if kind is not None:
            flats.append(FlatMoment(index=i, kind=kind, speaker_id=t.speaker_id))
            seen.add(i)
    # Topic death: the last turn of a topic that ended within ``short_topic_turns`` turns.
    n = len(segments)
    i = 0
    while i < n:
        topic = segments[i]
        j = i
        while j + 1 < n and segments[j + 1] == topic:
            j += 1
        ended = j + 1 < n  # a topic still running at the end of the transcript did not die
        if topic != NO_TOPIC and ended and (j - i + 1) <= short_topic_turns and j not in seen:
            flats.append(FlatMoment(index=j, kind="topic_death", speaker_id=turns[j].speaker_id))
            seen.add(j)
        i = j + 1
    flats.sort(key=lambda f: f.index)
    return flats


def is_repair_turn(turn: Turn, label: QuestionType | None = None, content_words: int = 8) -> bool:
    asks = label in ASKING if label is not None else is_question(turn.text)
    return asks or word_count(turn.text) >= content_words


def find_repairs(
    turns: Sequence[Turn],
    flats: Sequence[FlatMoment],
    labels: Sequence[QuestionType] | None = None,
    window: int = 3,
    content_words: int = 8,
) -> list[FlatMoment]:
    """Attach the first repairing turn within ``window`` turns after each flat moment."""
    out: list[FlatMoment] = []
    for flat in flats:
        repaired: FlatMoment = flat
        for j in range(flat.index + 1, min(len(turns), flat.index + window + 1)):
            lab = labels[j] if labels is not None else None
            if is_repair_turn(turns[j], lab, content_words):
                repaired = FlatMoment(
                    index=flat.index,
                    kind=flat.kind,
                    speaker_id=flat.speaker_id,
                    repaired_at=j,
                    repaired_by=turns[j].speaker_id,
                )
                break
        out.append(repaired)
    return out


def analyze_repairs(
    transcript: Transcript,
    segments: Sequence[int] | None = None,
    labels: Sequence[QuestionType] | None = None,
    window: int = 3,
) -> list[FlatMoment]:
    turns = transcript.spoken()
    return find_repairs(turns, detect_flat_moments(turns, segments), labels, window=window)


def repair_rate_from(flats: Sequence[FlatMoment]) -> float:
    """Repaired / flat moments. With nothing flat there was nothing to repair: 1.0."""
    if not flats:
        return 1.0
    return sum(f.repaired for f in flats) / len(flats)


def repair_latency_from(flats: Sequence[FlatMoment]) -> float:
    """Mean turns from flat moment to repair over repaired flats; 0.0 when none were repaired."""
    lat = [f.latency_turns for f in flats if f.latency_turns is not None]
    return sum(lat) / len(lat) if lat else 0.0


def repair_by_speaker_from(
    participants: Sequence[str], flats: Sequence[FlatMoment]
) -> dict[str, int]:
    counts = dict.fromkeys(participants, 0)
    for f in flats:
        if f.repaired_by is not None:
            counts[f.repaired_by] = counts.get(f.repaired_by, 0) + 1
    return counts


def repair_rate(transcript: Transcript, segments: Sequence[int] | None = None) -> float:
    return repair_rate_from(analyze_repairs(transcript, segments))


def repair_latency_turns(transcript: Transcript, segments: Sequence[int] | None = None) -> float:
    return repair_latency_from(analyze_repairs(transcript, segments))


def repair_by_speaker(
    transcript: Transcript, segments: Sequence[int] | None = None
) -> dict[str, int]:
    """Who repairs: repairs credited to the speaker of the repairing turn."""
    return repair_by_speaker_from(transcript.participants, analyze_repairs(transcript, segments))


__all__ = [
    "DEFLECTIONS",
    "FlatKind",
    "FlatMoment",
    "analyze_repairs",
    "detect_flat_moments",
    "find_repairs",
    "flat_kind",
    "is_deflection",
    "is_repair_turn",
    "repair_by_speaker",
    "repair_by_speaker_from",
    "repair_latency_from",
    "repair_latency_turns",
    "repair_rate",
    "repair_rate_from",
]
