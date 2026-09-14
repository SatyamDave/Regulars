import math

import pytest

from regulars.core.features import FOLLOW_UP_RATE, QUESTION_GINI, PairFeatures
from regulars.judge.diagnostics import (
    AUC_BROKEN,
    KnownPair,
    auc_on_known_pairs,
    known_pairs_report,
    variance_check,
)
from regulars.judge.v1 import JudgeV1


def test_variance_check_flags_constant_y_hat() -> None:
    report = variance_check([0.8] * 20)
    assert report.degenerate
    assert report.std == pytest.approx(0.0, abs=1e-12)
    assert report.iqr == pytest.approx(0.0, abs=1e-12)
    assert report.n == 20


def test_variance_check_passes_spread_scores() -> None:
    report = variance_check([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert not report.degenerate
    assert report.std == pytest.approx(math.sqrt(0.5))
    assert report.iqr == pytest.approx(1.0)
    assert variance_check([0.0, 0.01], floor=0.001).degenerate is False
    with pytest.raises(ValueError):
        variance_check([])


def test_auc_perfect_on_separable_data() -> None:
    assert auc_on_known_pairs([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auc_on_known_pairs([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert auc_on_known_pairs([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5


def test_auc_raises_on_single_class_or_bad_labels() -> None:
    with pytest.raises(ValueError, match="both a positive and a negative"):
        auc_on_known_pairs([0.1, 0.2], [1, 1])
    with pytest.raises(ValueError, match="0/1"):
        auc_on_known_pairs([0.1, 0.2], [1, 2])
    with pytest.raises(ValueError, match="same length"):
        auc_on_known_pairs([0.1], [1, 0])


def _known_pairs() -> list[KnownPair]:
    pairs = []
    # good pairs: many follow-ups, balanced questions; bad pairs: the reverse
    spec = [("a", "b", 0.5, 0.1, 1), ("c", "d", 0.4, 0.2, 1), ("e", "f", 0.1, 0.7, 0),
            ("g", "h", 0.05, 0.8, 0)]  # fmt: skip
    for a, b, fu, gini, label in spec:
        feats = {FOLLOW_UP_RATE: fu, QUESTION_GINI: gini}
        pairs.append(
            KnownPair(
                a=a,
                b=b,
                features_ab=PairFeatures(user_a=a, user_b=b, features=dict(feats)),
                features_ba=PairFeatures(user_a=b, user_b=a, features=dict(feats)),
                label=label,
            )
        )
    return pairs


def test_known_pairs_report_ranks_good_above_bad() -> None:
    pairs = _known_pairs()
    judge = JudgeV1.default([pf for p in pairs for pf in (p.features_ab, p.features_ba)])
    report = known_pairs_report(judge, pairs)
    assert report.auc == 1.0
    assert report.spearman > 0.8
    assert not report.variance.degenerate
    assert not report.broken
    assert [rp.label for rp in report.ranking] == [1, 1, 0, 0]
    assert report.ranking[0].score >= report.ranking[-1].score
    text = report.render()
    assert "AUC:         1.000" in text and "a <-> b" in text
    assert AUC_BROKEN == 0.6


def test_known_pairs_report_marks_degenerate_judge_broken() -> None:
    pairs = _known_pairs()
    judge = JudgeV1.default([pf for p in pairs for pf in (p.features_ab, p.features_ba)])
    zero = judge.model_copy(update={"weights": dict.fromkeys(judge.weights, 0.0)})
    report = known_pairs_report(zero, pairs)
    assert report.variance.degenerate
    assert report.broken
    assert math.isnan(report.spearman)
    assert "DEGENERATE" in report.render()
