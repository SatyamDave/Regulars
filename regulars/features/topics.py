"""Topic segmentation and topic survival (docs/technical-architecture.md §4).

A segmentation is one integer per spoken turn: the topic id (non-decreasing from 0), or
``NO_TOPIC`` (-1) for turns after a topic has died and before the next one starts. Long-tailed
turns-per-topic = flow; uniform short = interview mode.

``HeuristicTopicSegmenter`` is the lexical first pass; ``LLMTopicSegmenter`` segments all turns
in ONE call. Both satisfy ``TopicSegmenter``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from regulars.core.llm import LLMClient
from regulars.core.transcript import Transcript, Turn
from regulars.features._llm import batch_labels
from regulars.features._text import content_lemmas, jaccard
from regulars.features.questions import HeuristicQuestionClassifier, QuestionType

NO_TOPIC = -1


class TopicSegmenter(Protocol):
    """One topic id per spoken turn (``transcript.spoken()`` order); ``NO_TOPIC`` = dead air."""

    async def segment(self, transcript: Transcript) -> list[int]: ...


@dataclass(frozen=True)
class HeuristicTopicSegmenter:
    """Lexical segmentation.

    A turn starts a new topic when (a) it is a new-topic question, or (b) it carries content
    (>= ``min_content_words`` content lemmas) whose Jaccard overlap with the previous
    ``context_turns`` turns falls below ``threshold``, or (c) it carries content after the topic
    has died. Contentless turns ("yeah", "ok") ride the current topic but ``death_after``
    consecutive ones kill it; questions of any other type keep the topic alive.
    """

    threshold: float = 0.05
    context_turns: int = 3
    min_content_words: int = 2
    death_after: int = 2
    question_clf: HeuristicQuestionClassifier = field(default_factory=HeuristicQuestionClassifier)

    def segment_turns(
        self, turns: Sequence[Turn], labels: Sequence[QuestionType] | None = None
    ) -> list[int]:
        if labels is None:
            labels = self.question_clf.classify_turns(turns)
        lemmas = [content_lemmas(t.text) for t in turns]
        out: list[int] = []
        topic = NO_TOPIC
        next_id = 0
        contentless_run = 0
        for i, label in enumerate(labels):
            has_content = len(lemmas[i]) >= self.min_content_words
            if label == "new_topic":
                topic, next_id = next_id, next_id + 1
                contentless_run = 0
            elif label != "none":
                if topic == NO_TOPIC:
                    topic, next_id = next_id, next_id + 1
                contentless_run = 0
            elif not has_content:
                contentless_run += 1
                if topic != NO_TOPIC and contentless_run >= self.death_after:
                    out.append(topic)
                    topic = NO_TOPIC
                    continue
            else:
                contentless_run = 0
                context: set[str] = set()
                for j in range(max(0, i - self.context_turns), i):
                    context |= lemmas[j]
                if topic == NO_TOPIC or jaccard(lemmas[i], context) < self.threshold:
                    topic, next_id = next_id, next_id + 1
            out.append(topic)
        return out

    async def segment(self, transcript: Transcript) -> list[int]:
        return self.segment_turns(transcript.spoken())


_SYSTEM = """You segment a two-person conversation into topics. For each turn output the
integer id of the topic it belongs to: topics are numbered from 0 in order of appearance and a
new number means the turn starts a new subject. Output -1 for filler turns ("ok", "yeah") that
come after a topic has clearly died and before the next one begins. Ids never decrease except to
-1. Answer with a JSON array of integers only."""


@dataclass(frozen=True)
class LLMTopicSegmenter:
    """Segments every turn in one ``LLMClient`` call; bad entries fall back to the heuristic."""

    client: LLMClient
    fallback: HeuristicTopicSegmenter = field(default_factory=HeuristicTopicSegmenter)

    async def segment(self, transcript: Transcript) -> list[int]:
        turns = transcript.spoken()
        raw = await batch_labels(
            self.client,
            system=_SYSTEM,
            turns=turns,
            fallback=self.fallback.segment_turns(turns),
            normalize=_normalize,
            per_turn_tokens=4,
        )
        return normalize_segments(raw)


def _normalize(item: Any) -> int | None:
    if isinstance(item, bool):
        return None
    if isinstance(item, int):
        return item if item >= NO_TOPIC else None
    if isinstance(item, str) and item.strip().lstrip("-").isdigit():
        v = int(item.strip())
        return v if v >= NO_TOPIC else None
    return None


def normalize_segments(ids: Sequence[int]) -> list[int]:
    """Renumber topic ids to a dense, non-decreasing 0.. sequence; ``NO_TOPIC`` passes through.

    A model may skip numbers or reuse an earlier id; the statistics below only need contiguous
    runs, so each run of equal non-negative ids becomes its own topic.
    """
    out: list[int] = []
    next_id = 0
    prev: int | None = None
    for v in ids:
        if v == NO_TOPIC:
            out.append(NO_TOPIC)
            prev = None
            continue
        if prev is None or v != prev:
            out.append(next_id)
            next_id += 1
        else:
            out.append(out[-1])
        prev = v
    return out


def segment_lengths(segments: Sequence[int]) -> list[int]:
    """Turns per topic, in topic order (``NO_TOPIC`` turns excluded)."""
    counts = Counter(s for s in segments if s != NO_TOPIC)
    return [counts[k] for k in sorted(counts)]


def topic_survival(segments: Sequence[int], min_turns: int = 3) -> float:
    """Share of topics that lasted more than ``min_turns`` turns; 0.0 with no topics."""
    lengths = segment_lengths(segments)
    if not lengths:
        return 0.0
    return sum(n > min_turns for n in lengths) / len(lengths)


def topic_turns_p90(segments: Sequence[int]) -> float:
    """90th percentile of turns-per-topic (the long tail); 0.0 with no topics."""
    lengths = segment_lengths(segments)
    if not lengths:
        return 0.0
    return float(np.percentile(np.asarray(lengths, dtype=float), 90))


__all__ = [
    "NO_TOPIC",
    "HeuristicTopicSegmenter",
    "LLMTopicSegmenter",
    "TopicSegmenter",
    "normalize_segments",
    "segment_lengths",
    "topic_survival",
    "topic_turns_p90",
]
