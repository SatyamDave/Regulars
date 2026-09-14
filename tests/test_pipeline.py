"""``regulars.pipeline`` against a real SQLite engine and ``FakeLLM`` — no network, no judge
fitting on labels (technical-architecture.md §5: v1 is hand-set weights, not learned).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine

from regulars.core.features import FEATURE_NAMES
from regulars.core.llm import FakeLLM, LLMMessage
from regulars.core.persona import PersonaSpec, Social
from regulars.db import repo, schema
from regulars.db.engine import sqlite_memory_engine
from regulars.judge.v1 import JudgeV1
from regulars.pipeline import score_candidates
from regulars.sim.harness import SimConfig


def _responder(_system: str, messages: list[LLMMessage]) -> str:
    """A short, varied reply so the sim produces some feature variance across candidates."""
    last = messages[-1].content if messages else ""
    return f"That reminds me of something — what about you, way back when? ({len(last)})"


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = sqlite_memory_engine()
    yield eng
    eng.dispose()


@pytest.fixture
def user_and_candidates(engine: Engine) -> tuple[str, list[str]]:
    spec = PersonaSpec(social=Social(friction_points=["loud chewing"], drains=["small talk"]))
    with engine.begin() as conn:
        user = repo.create_user(conn, phone_hash="h-user", zip_code="94110")
        repo.append_persona_version(conn, user.id, spec, "voice_interview")
        candidates = []
        for i in range(3):
            c = repo.create_user(conn, phone_hash=f"h-c{i}", zip_code="94110")
            repo.append_persona_version(conn, c.id, spec, "voice_interview")
            candidates.append(c.id)
    return user.id, candidates


async def test_score_candidates_writes_sims_features_predictions_and_audit(
    engine: Engine, user_and_candidates: tuple[str, list[str]]
) -> None:
    user_id, candidate_ids = user_and_candidates
    llm = FakeLLM(responder=_responder)
    judge = JudgeV1.default([])
    sim_config = SimConfig(turns=2, seeds=(0,))

    predictions = await score_candidates(
        engine, user_id, llm, llm, judge, sim_config, candidate_ids=candidate_ids
    )

    assert {p.user_b for p in predictions} == set(candidate_ids)
    assert all(p.judge_version == judge.judge_version for p in predictions)

    with engine.begin() as conn:
        for candidate_id in candidate_ids:
            sims = conn.execute(
                schema.sims.select().where(
                    (schema.sims.c.user_a == user_id) & (schema.sims.c.user_b == candidate_id)
                )
            ).fetchall()
            assert sims  # at least one sim persisted per candidate

            sim_id = sims[0].id
            features = repo.sim_features_for(conn, sim_id)
            assert features  # some features were stored
            assert set(features) <= set(FEATURE_NAMES)

        audits = repo.audit_for_user(conn, user_id)
        scored = [a for a in audits if a.decision_type == "prediction_scored"]
        assert len(scored) == len(candidate_ids)
        for a in scored:
            assert a.judge_version == judge.judge_version
            assert a.persona_version_id is not None

        variance_rows = [a for a in audits if a.decision_type == "variance_check"]
        assert len(variance_rows) == 1
        report = variance_rows[0].payload
        assert report["n"] == len(candidate_ids)
        assert "std_y_hat" in report
        assert "mean_y_hat" in report


async def test_score_candidates_requires_a_persona(engine: Engine) -> None:
    with engine.begin() as conn:
        user = repo.create_user(conn, phone_hash="h-lonely", zip_code="94110")
    llm = FakeLLM(responder=_responder)
    judge = JudgeV1.default([])
    with pytest.raises(ValueError, match="no persona"):
        await score_candidates(engine, user.id, llm, llm, judge, candidate_ids=[])
