"""Feature extraction — one code path for simulated and real transcripts (architecture §4).

Every extractor takes a dyadic ``Transcript``; callers use ``Transcript.restrict(a, b)`` for
group chats. Judgment calls (question type, topic boundaries, disclosure depth) sit behind small
Protocols with a deterministic heuristic and an ``LLMClient``-backed implementation each.
"""

from regulars.features.disclosure import (
    DisclosureScorer,
    HeuristicDisclosureScorer,
    LLMDisclosureScorer,
    disclosure_asymmetry,
    disclosure_reciprocity,
)
from regulars.features.extract import (
    ExtractorConfig,
    extract_features_sync,
    extract_pair_features,
)
from regulars.features.lsm import (
    category_rates,
    lsm,
    lsm_mean,
    lsm_slope,
    lsm_trajectory,
)
from regulars.features.questions import (
    HeuristicQuestionClassifier,
    LLMQuestionClassifier,
    QuestionClassifier,
    QuestionType,
    follow_up_rate,
    question_gini,
)
from regulars.features.repair import (
    FlatMoment,
    repair_by_speaker,
    repair_latency_turns,
    repair_rate,
)
from regulars.features.topics import (
    HeuristicTopicSegmenter,
    LLMTopicSegmenter,
    TopicSegmenter,
    topic_survival,
    topic_turns_p90,
)

__all__ = [
    "DisclosureScorer",
    "ExtractorConfig",
    "FlatMoment",
    "HeuristicDisclosureScorer",
    "HeuristicQuestionClassifier",
    "HeuristicTopicSegmenter",
    "LLMDisclosureScorer",
    "LLMQuestionClassifier",
    "LLMTopicSegmenter",
    "QuestionClassifier",
    "QuestionType",
    "TopicSegmenter",
    "category_rates",
    "disclosure_asymmetry",
    "disclosure_reciprocity",
    "extract_features_sync",
    "extract_pair_features",
    "follow_up_rate",
    "lsm",
    "lsm_mean",
    "lsm_slope",
    "lsm_trajectory",
    "question_gini",
    "repair_by_speaker",
    "repair_latency_turns",
    "repair_rate",
    "topic_survival",
    "topic_turns_p90",
]
