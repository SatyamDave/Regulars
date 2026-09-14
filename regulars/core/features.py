"""Feature names and the per-pair feature container.

``FEATURE_NAMES`` is the registry the judge z-scores against. Add here first; extractors must
emit exactly these keys (real-only features are omitted, never zero-filled, for sim transcripts).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Shared by simulated and real transcripts.
FOLLOW_UP_RATE = "follow_up_rate"  # follow-up questions / speaker turns, directed (a asks b)
LSM_MEAN = "lsm_mean"  # Ireland & Pennebaker, mean over 9 function-word categories
LSM_SLOPE = "lsm_slope"  # per-turn LSM trajectory slope; convergence is the signal
TOPIC_SURVIVAL = "topic_survival"  # share of topics surviving past 3 turns
TOPIC_TURNS_P90 = "topic_turns_p90"  # long tail of turns-per-topic
DISCLOSURE_RECIPROCITY = "disclosure_reciprocity"  # lag-1 correlation of disclosure depth
DISCLOSURE_ASYMMETRY = "disclosure_asymmetry"  # |mean depth a - mean depth b|
REPAIR_RATE = "repair_rate"  # repaired flat moments / flat moments
REPAIR_LATENCY_TURNS = "repair_latency_turns"  # mean turns to recover
QUESTION_GINI = "question_gini"  # imbalance in who asks

# Real chat only. Never produced for sim transcripts.
LATENCY_NORM_MEDIAN = "latency_norm_median"  # directed, within-conversation normalized
LATENCY_NORM_FAST_SHARE = "latency_norm_fast_share"  # share of replies faster than own baseline
REPLY_WEIGHT = "reply_weight"  # directed reply-graph edge weight a->b
TAPBACK_RATE = "tapback_rate"  # directed tapbacks a->b per b message

SHARED_FEATURES: tuple[str, ...] = (
    FOLLOW_UP_RATE,
    LSM_MEAN,
    LSM_SLOPE,
    TOPIC_SURVIVAL,
    TOPIC_TURNS_P90,
    DISCLOSURE_RECIPROCITY,
    DISCLOSURE_ASYMMETRY,
    REPAIR_RATE,
    REPAIR_LATENCY_TURNS,
    QUESTION_GINI,
)
REAL_ONLY_FEATURES: tuple[str, ...] = (
    LATENCY_NORM_MEDIAN,
    LATENCY_NORM_FAST_SHARE,
    REPLY_WEIGHT,
    TAPBACK_RATE,
)
FEATURE_NAMES: tuple[str, ...] = SHARED_FEATURES + REAL_ONLY_FEATURES


class PairFeatures(BaseModel):
    """Directed features for the dyad a→b. Symmetric features carry the same value both ways."""

    user_a: str
    user_b: str
    features: dict[str, float] = Field(default_factory=dict)

    def get(self, name: str) -> float | None:
        return self.features.get(name)
