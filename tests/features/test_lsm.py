from __future__ import annotations

import pytest

from regulars.core.transcript import Transcript
from regulars.features.lsm import (
    ARTICLES,
    CATEGORIES,
    SMOOTHING,
    category_rates,
    lsm,
    lsm_by_category,
    lsm_mean,
    lsm_slope,
    lsm_trajectory,
)
from tests.features.conftest import A, B, dyad


def test_nine_categories_with_real_lexicons() -> None:
    assert len(CATEGORIES) == 9
    for name, lexicon in CATEGORIES.items():
        assert isinstance(lexicon, frozenset)
        if lexicon is ARTICLES:
            assert lexicon == {"a", "an", "the"}
        else:
            assert len(lexicon) >= 15, name


def test_category_rates_are_percentages() -> None:
    rates = category_rates("I am the one")
    assert rates["personal_pronouns"] == pytest.approx(25.0)
    assert rates["auxiliary_verbs"] == pytest.approx(25.0)
    assert rates["articles"] == pytest.approx(25.0)
    assert rates["negations"] == 0.0
    assert category_rates("") == dict.fromkeys(CATEGORIES, 0.0)


def test_lsm_matches_hand_computation() -> None:
    # a: "I am the one"     -> ppron 25, aux 25, art 25, others 0
    # b: "you are not here" -> ppron 25, aux 25, neg 25, adv 25, others 0
    a, b = "I am the one", "you are not here"
    per = lsm_by_category(a, b)
    same = 1.0  # 25 vs 25 -> 1 - 0/(50.0001)
    one_sided = 1.0 - 25.0 / (25.0 + SMOOTHING)  # 25 vs 0
    both_zero = 1.0  # 0 vs 0 -> 1 - 0/0.0001
    assert per["personal_pronouns"] == pytest.approx(same)
    assert per["auxiliary_verbs"] == pytest.approx(same)
    assert per["articles"] == pytest.approx(one_sided)
    assert per["negations"] == pytest.approx(one_sided)
    assert per["common_adverbs"] == pytest.approx(one_sided)
    for cat in ("impersonal_pronouns", "prepositions", "conjunctions", "quantifiers"):
        assert per[cat] == pytest.approx(both_zero)
    expected = (6 * 1.0 + 3 * one_sided) / 9  # 2 matched + 4 both-zero, 3 one-sided
    assert lsm(a, b) == pytest.approx(expected)


def test_lsm_identical_and_empty() -> None:
    text = "I really think that we should go to the park and not stay in."
    assert lsm(text, text) == pytest.approx(1.0)
    assert lsm(text, "") == 0.0
    assert lsm("", "") == 0.0


def test_lsm_mean_orders_fixtures(
    flow: Transcript, interview: Transcript, dead: Transcript
) -> None:
    assert lsm_mean(flow) > lsm_mean(interview)
    assert lsm_mean(flow) > lsm_mean(dead)


def test_trajectory_window_count(flow: Transcript) -> None:
    traj = lsm_trajectory(flow, window_turns=6)
    assert len(traj) == len(flow.spoken()) - 6 + 1
    assert all(0.0 <= v <= 1.0 for v in traj)
    short = dyad([(A, "I am here"), (B, "you are there")])
    assert len(lsm_trajectory(short, window_turns=6)) == 1
    with pytest.raises(ValueError):
        lsm_trajectory(flow, window_turns=1)


def test_trajectory_skips_single_speaker_windows() -> None:
    t = dyad([(A, "I am here")] * 6 + [(B, "you are there")] * 6)
    traj = lsm_trajectory(t, window_turns=6)
    assert len(traj) == 12 - 6 + 1 - 2  # the first and last windows are single-speaker


def test_slope_least_squares() -> None:
    assert lsm_slope([1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert lsm_slope([0.5, 0.5, 0.5]) == 0.0
    assert lsm_slope([0.7]) == 0.0
    assert lsm_slope([]) == 0.0
    assert lsm_slope([0.9, 0.6, 0.3]) < 0


def test_slope_positive_on_converging_transcript() -> None:
    # ana's style is fixed; ben starts with bare content words and gradually adopts her
    # function-word profile, so window LSM should rise over the conversation.
    ana = "I think it is the one for me and you, but we should not go there."
    ben_steps = [
        "Climbing. Weekend. Gym.",
        "Climbing at the gym.",
        "Climbing at the gym, not the park.",
        "I like climbing at the gym, not the park.",
        "I think climbing at the gym is the one for me.",
        "I think it is the one for me and you, but we should not go there.",
    ]
    lines = []
    for step in ben_steps:
        lines.extend([(A, ana), (B, step), (A, ana), (B, step)])
    t = dyad(lines)
    traj = lsm_trajectory(t, window_turns=4)
    assert lsm_slope(traj) > 0
    assert traj[-1] > traj[0]
