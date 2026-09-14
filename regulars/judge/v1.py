"""v1 judge: transparent, not learned (technical-architecture.md §5 "v1").

``y_hat = w · z(features)``. Weights are hand-set and signed; features are z-scored against a
``ZScaler`` fit on the population of ``PairFeatures`` the judge is applied to. Nothing here is
fit to labels on purpose: with more features than labeled dyads a learned model fits noise, and
knowing *which* signal carries the prediction is worth more than accuracy right now.

Every score returns its per-feature contributions so a ranking can be explained feature by
feature, and the judge serialises to JSON so ``judge_version`` names a reproducible function.
"""

from __future__ import annotations

import hashlib
import math

from pydantic import BaseModel, Field

from regulars.core.features import (
    DISCLOSURE_ASYMMETRY,
    DISCLOSURE_RECIPROCITY,
    FEATURE_NAMES,
    FOLLOW_UP_RATE,
    LATENCY_NORM_FAST_SHARE,
    LATENCY_NORM_MEDIAN,
    LSM_MEAN,
    LSM_SLOPE,
    QUESTION_GINI,
    REPAIR_RATE,
    REPLY_WEIGHT,
    TAPBACK_RATE,
    TOPIC_SURVIVAL,
    PairFeatures,
)

STD_FLOOR = 1e-6
"""A feature that is constant across the fit population gets this std so z stays finite (0)."""

# Signed weights on z-scored features. Magnitudes are comparable because every feature is
# z-scored; the sign encodes the direction the evidence points. Each entry cites the evidence
# (§4 of the architecture doc) that justifies its sign and rough size.
DEFAULT_WEIGHTS: dict[str, float] = {
    # + Huang, Yeomans, Brooks, Minson & Gino (JPSP 2017): follow-up questions predicted
    #   second-date agreement in speed dating; independently audited 2025, held. Strongest
    #   signal available to both sim and real transcripts, hence the largest positive weight.
    FOLLOW_UP_RATE: 1.0,
    # - Templeton et al. (PNAS 2022): faster responses predict enjoyment and connection at
    #   every level. The feature is log-latency relative to the user's own baseline, so
    #   *lower is better* and the weight is negative. Real chat only.
    LATENCY_NORM_MEDIAN: -1.0,
    # + Same evidence, distributional form: share of replies faster than the user's own
    #   baseline. Templeton's within-conversation finding; positive.
    LATENCY_NORM_FAST_SHARE: 0.75,
    # + Ireland & Pennebaker: convergence over the conversation is the signal, not level.
    #   Small because a slope estimated on a short chat is noisy.
    LSM_SLOPE: 0.5,
    # + Ireland et al. (2011) found LSM predicted interest in speed dating, but Niederhoffer &
    #   Pennebaker found it unrelated to rated interaction quality and read it as engagement,
    #   not rapport. Smaller still, and "feature, never target".
    LSM_MEAN: 0.25,
    # + Topics that survive past three turns are flow; uniform short topics are interview mode.
    TOPIC_SURVIVAL: 0.75,
    # + Lag-1 reciprocity of disclosure depth: mutual deepening, the closeness-generating
    #   pattern rather than one-sided confession.
    DISCLOSURE_RECIPROCITY: 0.75,
    # - One person carrying the disclosure is asymmetry, not intimacy.
    DISCLOSURE_ASYMMETRY: -0.5,
    # + Recovering from a flat moment is something both parties do together; this is what the
    #   ``dead_air`` scenario exists to elicit.
    REPAIR_RATE: 0.5,
    # - Gini over who asks: heavy imbalance means interview, not conversation.
    QUESTION_GINI: -0.5,
    # + Directed reply-graph weight a->b: the real social graph inside the group, which
    #   diverges sharply from the roster. Real chat only.
    REPLY_WEIGHT: 0.75,
    # + Tapbacks are a cheap, high-frequency, directed affect signal; small because cheap.
    TAPBACK_RATE: 0.25,
}

DEFAULT_VERSION = "v1.0-default"


class ZScaler(BaseModel):
    """Per-feature mean/std fit on a population of ``PairFeatures``.

    Features never observed in the fit population get ``mean=0, std=1`` and ``n=0``; any pair
    missing a feature is imputed to the mean (z = 0) and the name is reported in ``missing``.
    """

    mean: dict[str, float] = Field(default_factory=dict)
    std: dict[str, float] = Field(default_factory=dict)
    n: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def fit(cls, population: list[PairFeatures]) -> ZScaler:
        mean: dict[str, float] = {}
        std: dict[str, float] = {}
        n: dict[str, int] = {}
        for name in FEATURE_NAMES:
            values = [
                v
                for pf in population
                if (v := pf.features.get(name)) is not None and math.isfinite(v)
            ]
            n[name] = len(values)
            if not values:
                mean[name], std[name] = 0.0, 1.0
                continue
            mu = math.fsum(values) / len(values)
            var = math.fsum((v - mu) ** 2 for v in values) / len(values)
            mean[name] = mu
            std[name] = max(math.sqrt(var), STD_FLOOR)
        return cls(mean=mean, std=std, n=n)

    def transform(self, pf: PairFeatures) -> tuple[dict[str, float], list[str]]:
        """Return ``(z, missing)``; missing features are imputed to the mean (z = 0)."""
        z: dict[str, float] = {}
        missing: list[str] = []
        for name in self.mean:
            value = pf.features.get(name)
            if value is None or not math.isfinite(value):
                z[name] = 0.0
                missing.append(name)
            else:
                z[name] = (value - self.mean[name]) / self.std[name]
        return z, missing


class JudgeOutput(BaseModel):
    """One directed score with its explanation. ``sum(contributions.values()) == y_hat``."""

    y_hat: float
    contributions: dict[str, float]
    missing: list[str]


class JudgeV1(BaseModel):
    """Linear judge over z-scored features. Serialise with ``to_json`` to pin ``judge_version``."""

    weights: dict[str, float]
    scaler: ZScaler
    version: str = DEFAULT_VERSION

    def model_post_init(self, __context: object) -> None:
        unknown = set(self.weights) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"weights reference unknown features: {sorted(unknown)}")

    @classmethod
    def default(cls, population: list[PairFeatures], version: str = DEFAULT_VERSION) -> JudgeV1:
        """``DEFAULT_WEIGHTS`` with a scaler fit on ``population``."""
        return cls(weights=dict(DEFAULT_WEIGHTS), scaler=ZScaler.fit(population), version=version)

    def score(self, pf: PairFeatures) -> JudgeOutput:
        z, missing_all = self.scaler.transform(pf)
        contributions = {name: w * z.get(name, 0.0) for name, w in self.weights.items()}
        missing = [name for name in missing_all if name in self.weights]
        return JudgeOutput(
            y_hat=math.fsum(contributions.values()),
            contributions=contributions,
            missing=missing,
        )

    @property
    def judge_version(self) -> str:
        """``version`` plus a content hash, so two judges with the same label but different
        weights or scaler stay distinguishable in the ratings table."""
        return f"{self.version}@{self.fingerprint()}"

    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:12]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> JudgeV1:
        return cls.model_validate_json(text)


def pair_score(out_ab: JudgeOutput, out_ba: JudgeOutput) -> float:
    """Undirected pair score: mean of the two directed scores."""
    return (out_ab.y_hat + out_ba.y_hat) / 2.0
