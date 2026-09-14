"""Pair feature extraction: the one code path for sim and real transcripts.

``extract_pair_features`` runs the three judgment classifiers once each (concurrently), then
computes every ``SHARED_FEATURES`` key for both directions a->b and b->a. Directed features
(``follow_up_rate``: a asks b; ``disclosure_reciprocity``: a's depth vs b's reply) differ per
direction; the rest are symmetric and carried identically both ways. Real-only features are
never produced here — ``regulars.features.real`` merges them for timestamped chats.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from regulars.core.features import (
    DISCLOSURE_ASYMMETRY,
    DISCLOSURE_RECIPROCITY,
    FOLLOW_UP_RATE,
    LSM_MEAN,
    LSM_SLOPE,
    QUESTION_GINI,
    REPAIR_LATENCY_TURNS,
    REPAIR_RATE,
    SHARED_FEATURES,
    TOPIC_SURVIVAL,
    TOPIC_TURNS_P90,
    PairFeatures,
)
from regulars.core.llm import LLMClient
from regulars.core.transcript import Transcript
from regulars.features.disclosure import (
    DisclosureScorer,
    HeuristicDisclosureScorer,
    LLMDisclosureScorer,
    asymmetry_from_depths,
    reciprocity_from_depths,
)
from regulars.features.lsm import lsm_mean, lsm_slope, lsm_trajectory
from regulars.features.questions import (
    HeuristicQuestionClassifier,
    LLMQuestionClassifier,
    QuestionClassifier,
    follow_up_rate_from_labels,
    question_gini,
)
from regulars.features.repair import (
    analyze_repairs,
    repair_latency_from,
    repair_rate_from,
)
from regulars.features.topics import (
    HeuristicTopicSegmenter,
    LLMTopicSegmenter,
    TopicSegmenter,
    topic_survival,
    topic_turns_p90,
)


@dataclass(frozen=True)
class ExtractorConfig:
    """Which implementation answers each judgment call, plus the LSM window."""

    question_clf: QuestionClassifier
    segmenter: TopicSegmenter
    disclosure_scorer: DisclosureScorer
    lsm_window_turns: int = 6

    @classmethod
    def heuristic(cls, lsm_window_turns: int = 6) -> ExtractorConfig:
        """Deterministic regex/lexicon classifiers: tests and the cheap first pass."""
        return cls(
            question_clf=HeuristicQuestionClassifier(),
            segmenter=HeuristicTopicSegmenter(),
            disclosure_scorer=HeuristicDisclosureScorer(),
            lsm_window_turns=lsm_window_turns,
        )

    @classmethod
    def llm(cls, client: LLMClient, lsm_window_turns: int = 6) -> ExtractorConfig:
        """LLM-backed classifiers: three batched calls per transcript, heuristics as fallback."""
        return cls(
            question_clf=LLMQuestionClassifier(client),
            segmenter=LLMTopicSegmenter(client),
            disclosure_scorer=LLMDisclosureScorer(client),
            lsm_window_turns=lsm_window_turns,
        )


async def extract_pair_features(
    transcript: Transcript, a: str, b: str, config: ExtractorConfig
) -> tuple[PairFeatures, PairFeatures]:
    """Features for the dyad (a, b), returned as (a->b, b->a). Emits exactly SHARED_FEATURES."""
    if a == b:
        raise ValueError("a dyad needs two distinct participants")
    for who in (a, b):
        if who not in transcript.participants:
            raise ValueError(f"{who!r} is not a participant")
    dyad = transcript if transcript.participants == [a, b] else transcript.restrict(a, b)
    turns = dyad.spoken()

    labels, segments, depths = await asyncio.gather(
        config.question_clf.classify(dyad),
        config.segmenter.segment(dyad),
        config.disclosure_scorer.score(dyad),
    )
    flats = analyze_repairs(dyad, segments, labels)

    symmetric: dict[str, float] = {
        LSM_MEAN: lsm_mean(dyad),
        LSM_SLOPE: lsm_slope(lsm_trajectory(dyad, config.lsm_window_turns)),
        TOPIC_SURVIVAL: topic_survival(segments),
        TOPIC_TURNS_P90: topic_turns_p90(segments),
        DISCLOSURE_ASYMMETRY: asymmetry_from_depths(turns, depths, a, b),
        REPAIR_RATE: repair_rate_from(flats),
        REPAIR_LATENCY_TURNS: repair_latency_from(flats),
        QUESTION_GINI: question_gini(dyad, labels),
    }

    def directed(x: str, y: str) -> PairFeatures:
        feats = dict(symmetric)
        feats[FOLLOW_UP_RATE] = follow_up_rate_from_labels(turns, labels, x)
        feats[DISCLOSURE_RECIPROCITY] = reciprocity_from_depths(turns, depths, x, y)
        assert set(feats) == set(SHARED_FEATURES)
        return PairFeatures(user_a=x, user_b=y, features=feats)

    return directed(a, b), directed(b, a)


def extract_features_sync(
    transcript: Transcript, a: str, b: str, config: ExtractorConfig | None = None
) -> tuple[PairFeatures, PairFeatures]:
    """Blocking wrapper for the heuristic path (no event loop may be running)."""
    cfg = config or ExtractorConfig.heuristic()
    return asyncio.run(extract_pair_features(transcript, a, b, cfg))


__all__ = ["ExtractorConfig", "extract_features_sync", "extract_pair_features"]
