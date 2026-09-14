from __future__ import annotations

import math

import pytest

from regulars.core.features import (
    DISCLOSURE_RECIPROCITY,
    FOLLOW_UP_RATE,
    QUESTION_GINI,
    REAL_ONLY_FEATURES,
    REPAIR_RATE,
    SHARED_FEATURES,
    TOPIC_SURVIVAL,
    PairFeatures,
)
from regulars.core.llm import FakeLLM
from regulars.core.transcript import Transcript, Turn
from regulars.features.extract import (
    ExtractorConfig,
    extract_features_sync,
    extract_pair_features,
)
from tests.features.conftest import A, B
from tests.features.helpers import scripted_responder

cfg = ExtractorConfig.heuristic()


def _keys(pf: PairFeatures) -> set[str]:
    return set(pf.features)


async def test_emits_exactly_shared_features_both_directions(flow: Transcript) -> None:
    ab, ba = await extract_pair_features(flow, A, B, cfg)
    assert _keys(ab) <= set(SHARED_FEATURES)
    assert _keys(ab) == set(SHARED_FEATURES)
    assert _keys(ab) == _keys(ba)
    assert not (_keys(ab) & set(REAL_ONLY_FEATURES))
    assert (ab.user_a, ab.user_b) == (A, B)
    assert (ba.user_a, ba.user_b) == (B, A)
    assert all(math.isfinite(v) for v in ab.features.values())


async def test_directed_vs_symmetric(flow: Transcript) -> None:
    ab, ba = await extract_pair_features(flow, A, B, cfg)
    directed = {FOLLOW_UP_RATE, DISCLOSURE_RECIPROCITY}
    for k in SHARED_FEATURES:
        if k not in directed:
            assert ab.features[k] == ba.features[k], k
    assert ab.features[FOLLOW_UP_RATE] != ba.features[FOLLOW_UP_RATE]


async def test_fixture_orderings_end_to_end(
    flow: Transcript, interview: Transcript, dead: Transcript
) -> None:
    f_ab, f_ba = await extract_pair_features(flow, A, B, cfg)
    i_ab, i_ba = await extract_pair_features(interview, A, B, cfg)
    d_ab, _ = await extract_pair_features(dead, A, B, cfg)
    fu = FOLLOW_UP_RATE
    assert min(f_ab.features[fu], f_ba.features[fu]) > min(i_ab.features[fu], i_ba.features[fu])
    assert i_ab.features[QUESTION_GINI] > f_ab.features[QUESTION_GINI]
    assert f_ab.features[TOPIC_SURVIVAL] > d_ab.features[TOPIC_SURVIVAL]
    assert f_ab.features[TOPIC_SURVIVAL] > i_ab.features[TOPIC_SURVIVAL]
    assert f_ab.features[REPAIR_RATE] > d_ab.features[REPAIR_RATE]
    assert f_ab.features[DISCLOSURE_RECIPROCITY] > i_ab.features[DISCLOSURE_RECIPROCITY]


async def test_group_chat_is_restricted_to_the_dyad(flow: Transcript) -> None:
    turns = list(flow.turns)
    turns.insert(3, Turn(speaker_id="cy", text="lol"))
    turns.append(Turn(speaker_id="cy", text="anyway who wants pizza?"))
    group = Transcript(participants=[A, B, "cy"], turns=turns, source="real")
    g_ab, _ = await extract_pair_features(group, A, B, cfg)
    f_ab, _ = await extract_pair_features(flow, A, B, cfg)
    assert g_ab.features == f_ab.features


async def test_real_source_uses_the_same_path(flow: Transcript) -> None:
    real = Transcript(
        participants=flow.participants,
        turns=[t.model_copy(update={"sent_at_ms": 1000 * i}) for i, t in enumerate(flow.turns)],
        source="real",
    )
    r_ab, _ = await extract_pair_features(real, A, B, cfg)
    f_ab, _ = await extract_pair_features(flow, A, B, cfg)
    assert r_ab.features == f_ab.features


async def test_llm_config_makes_three_batched_calls(flow: Transcript) -> None:
    fake = FakeLLM(responder=scripted_responder)
    ab, ba = await extract_pair_features(flow, A, B, ExtractorConfig.llm(fake))
    assert len(fake.calls) == 3
    assert _keys(ab) == set(SHARED_FEATURES) == _keys(ba)
    assert ab.features[FOLLOW_UP_RATE] == 0.0  # every turn scripted as "none"
    assert ab.features[TOPIC_SURVIVAL] == 0.0  # every turn its own topic


def test_sync_wrapper_matches_async(flow: Transcript) -> None:
    ab, ba = extract_features_sync(flow, A, B)
    assert _keys(ab) == set(SHARED_FEATURES) == _keys(ba)
    assert ab.features[FOLLOW_UP_RATE] != ba.features[FOLLOW_UP_RATE]


async def test_validation_errors(flow: Transcript) -> None:
    with pytest.raises(ValueError):
        await extract_pair_features(flow, A, A, cfg)
    with pytest.raises(ValueError):
        await extract_pair_features(flow, A, "nobody", cfg)


async def test_empty_transcript_does_not_crash() -> None:
    empty = Transcript(participants=[A, B], turns=[], source="sim")
    ab, ba = await extract_pair_features(empty, A, B, cfg)
    assert _keys(ab) == set(SHARED_FEATURES) == _keys(ba)
    assert all(math.isfinite(v) for v in ab.features.values())
