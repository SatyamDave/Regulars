"""Offline batch orchestration: the ten subpackages' domain logic, driven with injected LLMs and
a SQLAlchemy ``Engine``, so it runs against ``FakeLLM`` and SQLite in tests with no network
(technical-architecture.md §0, §9).

Like ``regulars.wiring``, this module is the integration layer allowed to import across every
subpackage. Four entry points, one per step of the loop in §0:

- :func:`score_candidates` — persona × persona sim → features → judge → stored predictions,
  for one user against everyone plausible.
- :func:`assemble_and_open` — predictions → group assembly (with the exploration arm) →
  ``groups``/``memberships`` rows. Returns the opened groups; the *caller* drives
  ``regulars.bot.flows.GroupOpener`` (this module never touches a ``Transport``).
- :func:`post_dinner` — ratings + attendance → prune/backfill → updated memberships, and folds
  observed chat behavior into a ``chat_behavior`` persona version when there is chat to observe.
- :func:`legibility_reply` / :func:`apply_persona_correction` — the "what do you know about me"
  loop (§10).

Every decision writes an audit row (§9): this module never mutates ``groups``, ``memberships``,
`persona_versions`` or drops a member without one.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import Connection, Engine

from regulars.assembly.exploration import (
    DEFAULT_EXPLORATION_RATE,
    assemble_round,
)
from regulars.assembly.optimizer import PairScores
from regulars.assembly.prefilter import Pair, pair_key
from regulars.assembly.rounds import (
    AttendanceRecord,
    DirectedRating,
    RoundUpdate,
    prune_and_backfill,
)
from regulars.core.features import PairFeatures
from regulars.core.llm import LLMClient
from regulars.core.persona import Conversational, PersonaSpec, Social
from regulars.db import repo
from regulars.db.schema import AssemblyMethod
from regulars.features.extract import ExtractorConfig, extract_pair_features
from regulars.judge.v1 import JudgeV1, pair_score
from regulars.persona import legibility
from regulars.persona.versioning import merge_update
from regulars.sim.harness import SimConfig, run_pair


@dataclass(frozen=True)
class OpenedGroup:
    """One assembled table, ready for ``regulars.bot.flows.GroupOpener.open``."""

    group_id: str
    members: tuple[str, ...]
    assembly_method: AssemblyMethod
    min_score: float


# --- score_candidates ----------------------------------------------------------------------


def _average_features(features: Sequence[PairFeatures]) -> PairFeatures:
    if not features:
        raise ValueError("no features to average")
    keys = features[0].features.keys()
    merged = {k: statistics.fmean(f.features[k] for f in features) for k in keys}
    return PairFeatures(user_a=features[0].user_a, user_b=features[0].user_b, features=merged)


def _candidate_pool(conn: Connection, user_id: str) -> list[repo.User]:
    """Everyone but ``user_id`` who shares their zip and has at least one persona version.

    A full prefilter (technical-architecture.md §7.1) also checks availability and stated
    exclusions; neither is modelled in the §1 schema, so zip compatibility is the only hard
    constraint applied here. Widening this is a schema addition, not a pipeline change.
    """
    me = repo.get_user(conn, user_id)
    out = []
    for candidate in repo.list_users(conn, zip_code=me.zip):
        if candidate.id == user_id:
            continue
        if repo.latest_persona_version(conn, candidate.id) is not None:
            out.append(candidate)
    return out


async def score_candidates(
    engine: Engine,
    user_id: str,
    llm_cheap: LLMClient,
    llm_frontier: LLMClient,
    judge: JudgeV1,
    sim_config: SimConfig | None = None,
    *,
    candidate_ids: Sequence[str] | None = None,
    extractor: ExtractorConfig | None = None,
) -> list[repo.Prediction]:
    """Score ``user_id`` against every plausible candidate; persist sims/features/predictions.

    ``llm_cheap`` is accepted for the §8 cost-model seam (a two-stage cheap-then-frontier pass)
    but the frontier model runs the whole ``sim_config`` grid here; swap in
    ``regulars.sim.two_stage.rank_candidates`` for the full two-stage pipeline once its
    synchronous ``ScoreFn`` is fed pre-extracted features. Writes one ``prediction_scored`` audit
    row per candidate and one ``variance_check`` row (§6 Problem 4: watch variance before mean).
    """
    sim_config = sim_config or SimConfig()
    extractor = extractor or ExtractorConfig.heuristic()

    with engine.begin() as conn:
        persona_a = repo.latest_persona_version(conn, user_id)
        if persona_a is None:
            raise ValueError(f"{user_id} has no persona version to score against")
        if candidate_ids is not None:
            candidates = [repo.get_user(conn, cid) for cid in candidate_ids]
        else:
            candidates = _candidate_pool(conn, user_id)
        personas_b = {c.id: repo.latest_persona_version(conn, c.id) for c in candidates}

    predictions: list[repo.Prediction] = []
    for candidate in candidates:
        persona_b = personas_b[candidate.id]
        if persona_b is None:
            continue
        transcripts = await run_pair(
            llm_frontier,
            llm_frontier,
            persona_a.spec,
            persona_b.spec,
            sim_config,
            participants=(user_id, candidate.id),
        )
        ab_features: list[PairFeatures] = []
        ba_features: list[PairFeatures] = []
        with engine.begin() as conn:
            for transcript in transcripts:
                ab, ba = await extract_pair_features(transcript, user_id, candidate.id, extractor)
                sim_row = repo.insert_sim(
                    conn,
                    user_a=user_id,
                    user_b=candidate.id,
                    persona_ver_a=persona_a.id,
                    persona_ver_b=persona_b.id,
                    scenario_id=str(transcript.metadata.get("scenario_id", "")),
                    seed=int(transcript.metadata.get("seed", 0)),
                    transcript=transcript,
                )
                repo.insert_sim_features(conn, sim_row.id, ab.features)
                ab_features.append(ab)
                ba_features.append(ba)

            out_ab = judge.score(_average_features(ab_features))
            out_ba = judge.score(_average_features(ba_features))
            y_hat = pair_score(out_ab, out_ba)
            prediction = repo.insert_prediction(
                conn,
                user_a=user_id,
                user_b=candidate.id,
                judge_version=judge.judge_version,
                y_hat=y_hat,
            )
            repo.write_audit(
                conn,
                actor="pipeline",
                decision_type="prediction_scored",
                reason=f"scored {user_id} x {candidate.id} over {len(transcripts)} sims",
                payload={
                    "y_hat": y_hat,
                    "persona_ver_a": persona_a.id,
                    "persona_ver_b": persona_b.id,
                    "n_sims": len(transcripts),
                },
                subject_user_id=user_id,
                persona_version_id=persona_a.id,
                judge_version=judge.judge_version,
            )
        predictions.append(prediction)

    y_hats = [p.y_hat for p in predictions]
    variance_report = {
        "n": len(y_hats),
        "std_y_hat": statistics.pstdev(y_hats) if len(y_hats) > 1 else 0.0,
        "mean_y_hat": statistics.fmean(y_hats) if y_hats else 0.0,
        "non_degenerate": (statistics.pstdev(y_hats) if len(y_hats) > 1 else 0.0) > 1e-9,
    }
    with engine.begin() as conn:
        repo.write_audit(
            conn,
            actor="pipeline",
            decision_type="variance_check",
            reason=(
                "§6 Problem 4 diagnostic: watch variance before mean. "
                f"std(y_hat)={variance_report['std_y_hat']:.4f} over {variance_report['n']} pairs."
            ),
            payload=variance_report,
            subject_user_id=user_id,
            judge_version=judge.judge_version,
        )
    return predictions


# --- assemble_and_open ---------------------------------------------------------------------


def assemble_and_open(
    engine: Engine,
    zip_code: str,
    rng: random.Random,
    exploration_rate: float = DEFAULT_EXPLORATION_RATE,
    *,
    group_size: int = 6,
    judge_version: str | None = None,
    previous_groups: Sequence[Sequence[str]] = (),
) -> list[OpenedGroup]:
    """Assemble this round's tables for ``zip_code`` from the latest predictions.

    Plausibility here is "both are active users in this zip"; see :func:`_candidate_pool`'s
    docstring for the same caveat about availability/exclusions not being in the §1 schema.
    """
    with engine.begin() as conn:
        users = [u.id for u in repo.list_users(conn, zip_code=zip_code, status="active")]
        preds = repo.latest_predictions_for(conn, users, judge_version=judge_version)
        scores = PairScores()
        for p in preds:
            y_hat = p.y_hat_calibrated if p.y_hat_calibrated is not None else p.y_hat
            scores.set(p.user_a, p.user_b, y_hat)
        plausible: frozenset[Pair] = frozenset(
            pair_key(a, b) for i, a in enumerate(users) for b in users[i + 1 :]
        )
        plans = assemble_round(
            users,
            scores,
            plausible,
            previous_groups,
            rng,
            exploration_rate,
            group_size=group_size,
        )
        opened: list[OpenedGroup] = []
        for plan in plans:
            group = repo.create_group(conn, zip_code=zip_code, assembly_method=plan.assembly_method)
            for user_id in plan.members:
                repo.add_membership(conn, group.id, user_id, joined_round=1)
            for record in plan.audit:
                repo.write_audit(
                    conn,
                    actor="pipeline",
                    decision_type=record.decision_type,
                    reason=record.reason,
                    payload=record.payload,
                    subject_user_id=record.subject_user_id,
                    group_id=group.id,
                )
            opened.append(
                OpenedGroup(
                    group_id=group.id,
                    members=plan.members,
                    assembly_method=plan.assembly_method,
                    min_score=plan.min_score,
                )
            )
        return opened


# --- post_dinner ---------------------------------------------------------------------------


_SHORT_CHARS = 40
_LONG_CHARS = 120


def _observed_spec_from_messages(
    user_char_counts: Sequence[int], user_count: int, avg_count: float
) -> PersonaSpec:
    """A minimal chat-behavior observation from what §1's ``messages`` table actually carries:
    message length and share of the group's traffic. Real behavioral inference (§4 features:
    follow-up rate, LSM, disclosure, latency) belongs in a richer version of this function once
    a feature->persona mapping exists; this is deliberately conservative rather than invented.
    """
    avg_len = statistics.fmean(user_char_counts) if user_char_counts else 0.0
    avg_turn_length: Literal["short", "medium", "long"]
    if avg_len < _SHORT_CHARS:
        avg_turn_length = "short"
    elif avg_len > _LONG_CHARS:
        avg_turn_length = "long"
    else:
        avg_turn_length = "medium"

    energy: Literal["initiator", "responder", "observer"]
    if avg_count > 0 and user_count > avg_count * 1.2:
        energy = "initiator"
    elif avg_count > 0 and user_count < avg_count * 0.8:
        energy = "observer"
    else:
        energy = "responder"
    return PersonaSpec(
        conversational=Conversational(avg_turn_length=avg_turn_length),
        social=Social(energy=energy),
    )


def _apply_chat_behavior_update(conn: Connection, group_id: str) -> None:
    messages = repo.messages_for_group(conn, group_id)
    if not messages:
        return
    by_user: dict[str, list[int]] = {}
    for m in messages:
        by_user.setdefault(m.user_id, []).append(m.char_count)
    avg_count = statistics.fmean(len(v) for v in by_user.values())
    for user_id, char_counts in by_user.items():
        base = repo.latest_persona_version(conn, user_id)
        if base is None:
            continue
        observed = _observed_spec_from_messages(char_counts, len(char_counts), avg_count)
        merged = merge_update(base.spec, observed, "chat_behavior")
        if merged == base.spec:
            continue
        new_version = repo.append_persona_version(conn, user_id, merged, "chat_behavior")
        repo.write_audit(
            conn,
            actor="pipeline",
            decision_type="persona_updated",
            reason="folded observed chat behavior into the persona (source=chat_behavior)",
            payload={"n_messages": len(char_counts)},
            subject_user_id=user_id,
            group_id=group_id,
            persona_version_id=new_version.id,
        )


def post_dinner(
    engine: Engine,
    plan_id: str,
    *,
    queue: Sequence[str] = (),
    scores: PairScores | None = None,
    min_keep: int = 3,
) -> RoundUpdate:
    """Prune, backfill, and fold observed chat behavior into personas, after one dinner.

    Returns the ``regulars.assembly.rounds.RoundUpdate``. See the module docstring: removals and
    additions each write an audit row with a reason fit to show the affected user (§10).
    """
    with engine.begin() as conn:
        plan = repo.get_plan(conn, plan_id)
        group = repo.get_group(conn, plan.group_id)
        current = repo.memberships_for_group(conn, group.id, active_only=True)
        group_members = tuple(m.user_id for m in current)
        round_no = max((m.joined_round for m in current), default=0) + 1

        ratings = [
            DirectedRating(rater=r.rater_id, ratee=r.ratee_id, would_see_again=r.would_see_again)
            for r in repo.ratings_for_plan(conn, plan_id)
        ]
        paid = {r.user_id for r in repo.rsvps_for_plan(conn, plan_id) if r.state == "paid"}
        attendance = [
            AttendanceRecord(user_id=a.user_id, attended=a.attended, paid=a.user_id in paid)
            for a in repo.attendance_for_plan(conn, plan_id)
        ]

        update = prune_and_backfill(
            group_members, ratings, attendance, list(queue), scores or PairScores(), min_keep
        )
        for removal in update.removed:
            repo.end_membership(
                conn, group.id, removal.user_id, left_round=round_no, exit_reason=removal.reason
            )
        for user_id in update.added:
            repo.add_membership(conn, group.id, user_id, joined_round=round_no)
        for record in update.audit:
            repo.write_audit(
                conn,
                actor="pipeline",
                decision_type=record.decision_type,
                reason=record.reason,
                payload=record.payload,
                subject_user_id=record.subject_user_id,
                group_id=group.id,
                plan_id=plan_id,
            )
        _apply_chat_behavior_update(conn, group.id)
        return update


# --- legibility (§10) ------------------------------------------------------------------------


def legibility_reply(engine: Engine, user_id: str) -> str:
    """ "What do you know about me", rendered plain English (§10)."""
    with engine.begin() as conn:
        persona = repo.latest_persona_version(conn, user_id)
    if persona is None:
        return legibility.EMPTY
    return legibility.render_plain_english(persona.spec)


def apply_persona_correction(
    engine: Engine, user_id: str, line_number: int, new_value: Any
) -> repo.PersonaVersion:
    """Turn "line N is wrong, it's actually X" into a new, corrected persona version."""
    with engine.begin() as conn:
        persona = repo.latest_persona_version(conn, user_id)
        if persona is None:
            raise ValueError(f"{user_id} has no persona to correct")
        numbered = legibility.numbered_fields(persona.spec)
        match = next((f for f in numbered if f[0] == line_number), None)
        if match is None:
            raise ValueError(f"no field at line {line_number}")
        _, field_path, _ = match
        corrected = legibility.apply_correction(persona.spec, field_path, new_value)
        return repo.append_persona_version(conn, user_id, corrected, "user_correction")


__all__ = [
    "OpenedGroup",
    "apply_persona_correction",
    "assemble_and_open",
    "legibility_reply",
    "post_dinner",
    "score_candidates",
]
