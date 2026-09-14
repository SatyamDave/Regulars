"""``python -m regulars.ops.seed`` — build a realistic-looking demo database.

There is otherwise no way to spin up a browsable, internally-consistent local database: every
test uses minimal fixtures scoped to that one test. This script drives the real domain functions
(never raw inserts) to produce one that looks like a live "Regulars" launch zip
(docs/product-plan.md §9: "Week 1: 60 individual texts, one zip code"):

1. ``N`` users in one zip code, plausible fake ``phone_hash`` values.
2. A varied, internally-consistent ``PersonaSpec`` per user (source ``voice_interview``), each
   carrying real ``friction_points``/``drains`` (core.persona: "load-bearing for the simulator"),
   appended through :func:`regulars.persona.versioning.next_version` so it goes through the same
   append-only path a real voice interview would.
3. A default judge (:meth:`regulars.judge.v1.JudgeV1.default`) fit on synthetic
   :class:`~regulars.core.features.PairFeatures` for the hard-constraint-prefiltered candidate set
   (:func:`regulars.assembly.prefilter.plausible_pairs`); one ``sims``/``sim_features`` row and one
   ``predictions`` row per pair, via the repository functions real code uses.
4. One round of :func:`regulars.assembly.exploration.assemble_round` (the real 30% exploration
   split), writing ``groups``/``memberships``/``audit_log`` rows.
5. A handful of those groups get a simulated completed dinner (varied directed ratings including
   at least one clearly-unpopular member, plus one paid no-show) and then an actual call to
   :func:`regulars.pipeline.post_dinner`, so the seeded database has a real removal-with-reason
   audit trail to click through, not a synthesized one.

Safety rail: refuses to run against a non-empty database (any existing ``users`` row) unless
``--force`` is passed, so nobody accidentally seeds fake data on top of a real ``DATABASE_URL``.
Not a true idempotent seeder — rerunning with ``--force`` against the same file adds another
batch of (colliding, since phone_hash is deterministic per seed+index) users rather than
replacing anything; the intended dev workflow is deleting ``dev.db`` and reseeding.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import Connection, Engine, Table, func, select

from regulars.assembly.exploration import DEFAULT_EXPLORATION_RATE, assemble_round
from regulars.assembly.optimizer import PairScores
from regulars.assembly.prefilter import Candidate, plausible_pairs
from regulars.core.features import PairFeatures
from regulars.core.persona import Content, Conversational, PersonaSpec, Provenance, Social, Stated
from regulars.core.transcript import Transcript, Turn
from regulars.db import repo, schema
from regulars.db.engine import make_engine
from regulars.judge.v1 import JudgeV1, pair_score
from regulars.persona import versioning
from regulars.pipeline import post_dinner

DEFAULT_DB_PATH = Path("dev.db")
DEFAULT_ZIP = "94110"
DEFAULT_USERS = 60
GROUP_SIZE = 6
MAX_PARTNERS_PER_USER = 200
"""Cap passed to ``plausible_pairs`` (production's own default). The synthetic availability
below is deliberately generous (a 3-slot pool, 2-3 slots per user) so almost every same-zip pair
clears the hard-constraint prefilter; capping partners tighter than that would carve holes out of
what is otherwise a complete graph and could leave ``assemble_round`` unable to find a single
6-clique to seat, especially at the small ``--users`` counts used in tests."""
ROUNDS_TO_SIMULATE = 3
"""How many opened groups get a simulated completed dinner + ``post_dinner`` call."""

_TOPIC_POOL = (
    "cooking",
    "hiking",
    "startups",
    "board games",
    "live music",
    "true crime podcasts",
    "travel",
    "climbing",
    "photography",
    "local politics",
    "philosophy",
    "video games",
    "wine",
    "running",
)
_FRICTION_POOL = (
    "gets defensive when challenged",
    "dominates the conversation",
    "checks their phone constantly",
    "one-ups every story",
    "avoids eye contact",
    "interrupts often",
    "over-shares immediately",
    "goes quiet under any pressure",
)
_DRAIN_POOL = (
    "small talk",
    "large loud groups",
    "conflict",
    "unstructured plans",
    "being the center of attention",
    "long silences",
    "networking-flavored conversation",
)
_AVOIDED_POOL = ("exes", "religion", "salary", "family drama")
_ESCALATES_POOL = ("food", "travel stories", "music taste", "career pivots")
_DEFLECTS_POOL = ("personal finances", "past relationships", "health issues")
_EXCLUSION_POOL = ("smokers", "people who are chronically late", "very online people")
_OPENER_POOL = (
    "asks about your week",
    "jumps straight into a hot take",
    "compliments something specific",
    "makes a joke first",
)
_SEEKING_POOL = (
    "new friends in the city",
    "people to grab dinner with regularly",
    "a low-key social outlet",
    "folks with similar hobbies",
)
_SLOT_POOL = ("tue_dinner", "wed_dinner", "thu_dinner")
_VENUE_POOL = ("Bar Tartine", "Rich Table", "State Bird Provisions", "Lazy Bear", "Nopa")

_FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "follow_up_rate": (0.0, 1.0),
    "lsm_mean": (0.2, 0.9),
    "lsm_slope": (-0.1, 0.15),
    "topic_survival": (0.0, 1.0),
    "topic_turns_p90": (1.0, 10.0),
    "disclosure_reciprocity": (-1.0, 1.0),
    "disclosure_asymmetry": (0.0, 1.0),
    "repair_rate": (0.0, 1.0),
    "repair_latency_turns": (1.0, 5.0),
    "question_gini": (0.0, 1.0),
    "latency_norm_median": (-1.0, 1.0),
    "latency_norm_fast_share": (0.0, 1.0),
    "reply_weight": (0.0, 1.0),
    "tapback_rate": (0.0, 0.5),
}


@dataclass(frozen=True)
class SeedSummary:
    db_url: str
    n_users: int
    n_personas: int
    n_predictions: int
    n_groups: int
    n_rounds_simulated: int
    n_removed: int
    n_audit_rows: int

    def render(self) -> str:
        return (
            "Seeded demo database\n"
            f"  db url:            {self.db_url}\n"
            f"  users:             {self.n_users}\n"
            f"  persona versions:  {self.n_personas}\n"
            f"  predictions:       {self.n_predictions}\n"
            f"  groups:            {self.n_groups}\n"
            f"  rounds simulated:  {self.n_rounds_simulated}\n"
            f"  members removed:   {self.n_removed}\n"
            f"  audit rows:        {self.n_audit_rows}\n"
        )


def _phone_hash(seed: int, index: int) -> str:
    """A plausible-looking fake phone hash: sha256 of a synthetic +1 number, deterministic."""
    fake_number = f"+1555{seed % 1000:03d}{index:05d}"
    return hashlib.sha256(fake_number.encode()).hexdigest()


def _random_persona_spec(rng: random.Random) -> PersonaSpec:
    """A varied but internally-consistent spec; friction_points/drains are always non-empty
    (core.persona.PersonaSpec.has_negative_space) since a blank persona gives the simulator no
    variance to work with."""
    return PersonaSpec(
        conversational=Conversational(
            opener_style=rng.choice(_OPENER_POOL),
            avg_turn_length=rng.choice(["short", "medium", "long"]),
            humor_register=rng.choice(["dry", "absurdist", "self-deprecating", "none"]),
            question_tendency=rng.choice(["high", "low"]),
            interrupt_tendency=rng.choice(
                ["rarely interrupts", "frequently interrupts", "interrupts when excited"]
            ),
            silence_tolerance=rng.choice(
                ["comfortable with silence", "fills every gap immediately"]
            ),
        ),
        content=Content(
            unprompted_topics=rng.sample(_TOPIC_POOL, k=rng.randint(2, 4)),
            avoided_topics=rng.sample(_AVOIDED_POOL, k=rng.randint(0, 2)),
            escalates_on=rng.sample(_ESCALATES_POOL, k=rng.randint(1, 2)),
            deflects_on=rng.sample(_DEFLECTS_POOL, k=rng.randint(0, 2)),
        ),
        social=Social(
            energy=rng.choice(["initiator", "responder", "observer"]),
            disclosure_speed=rng.choice(["fast", "slow"]),
            friction_points=rng.sample(_FRICTION_POOL, k=rng.randint(1, 2)),
            drains=rng.sample(_DRAIN_POOL, k=rng.randint(1, 2)),
        ),
        stated=Stated(
            exclusions=rng.sample(_EXCLUSION_POOL, k=rng.randint(0, 1)),
            seeking=rng.choice(_SEEKING_POOL),
        ),
        provenance=Provenance(sources=["voice_interview"]),
    )


def _synthetic_features(rng: random.Random) -> dict[str, float]:
    return {name: rng.uniform(lo, hi) for name, (lo, hi) in _FEATURE_RANGES.items()}


def _seed_users_and_personas(
    conn: Connection, rng: random.Random, seed: int, n_users: int, zip_code: str
) -> tuple[list[repo.User], dict[str, repo.PersonaVersion]]:
    users: list[repo.User] = []
    personas: dict[str, repo.PersonaVersion] = {}
    for i in range(n_users):
        user = repo.create_user(conn, phone_hash=_phone_hash(seed, i), zip_code=zip_code)
        users.append(user)
        spec = _random_persona_spec(rng)
        # Route through the real append-only versioning path rather than a raw insert: build the
        # pure value with next_version, then persist it exactly as regulars.persona would.
        version = versioning.next_version([], spec, "voice_interview", user_id=user.id)
        stored = repo.append_persona_version(conn, user.id, version.spec, version.source)
        personas[user.id] = stored
    return users, personas


def _candidate_from_user(user: repo.User, rng: random.Random) -> Candidate:
    availability = frozenset(rng.sample(_SLOT_POOL, k=rng.randint(2, len(_SLOT_POOL))))
    return Candidate(user_id=user.id, zip=user.zip, availability=availability)


def _seed_sims_and_predictions(
    conn: Connection,
    rng: random.Random,
    seed: int,
    users: list[repo.User],
    personas: dict[str, repo.PersonaVersion],
) -> PairScores:
    candidates = [_candidate_from_user(u, rng) for u in users]
    pairs = sorted(plausible_pairs(candidates, max_per_user=MAX_PARTNERS_PER_USER, seed=seed))

    features_by_pair: dict[tuple[str, str], tuple[dict[str, float], dict[str, float]]] = {
        pair: (_synthetic_features(rng), _synthetic_features(rng)) for pair in pairs
    }
    population: list[PairFeatures] = []
    for (a, b), (fab, fba) in features_by_pair.items():
        population.append(PairFeatures(user_a=a, user_b=b, features=fab))
        population.append(PairFeatures(user_a=b, user_b=a, features=fba))
    judge = JudgeV1.default(population)

    scores = PairScores()
    for (a, b), (fab, fba) in features_by_pair.items():
        out_ab = judge.score(PairFeatures(user_a=a, user_b=b, features=fab))
        out_ba = judge.score(PairFeatures(user_a=b, user_b=a, features=fba))
        y_hat = pair_score(out_ab, out_ba)

        turns = [
            Turn(speaker_id=a, text="How was your week?"),
            Turn(speaker_id=b, text="Pretty good, actually — tell me about yours."),
        ]
        transcript = Transcript(
            participants=[a, b], turns=turns, source="sim", metadata={"scenario_id": "seed_dev"}
        )
        sim_row = repo.insert_sim(
            conn,
            user_a=a,
            user_b=b,
            persona_ver_a=personas[a].id,
            persona_ver_b=personas[b].id,
            scenario_id="seed_dev",
            seed=seed,
            transcript=transcript,
        )
        repo.insert_sim_features(conn, sim_row.id, fab)
        repo.insert_prediction(
            conn, user_a=a, user_b=b, judge_version=judge.judge_version, y_hat=y_hat
        )
        scores.set(a, b, y_hat)

    return scores


def _open_groups(
    conn: Connection, rng: random.Random, zip_code: str, users: list[repo.User], scores: PairScores
) -> list[repo.Group]:
    user_ids = [u.id for u in users]
    # Every pair we scored is, by construction, one that cleared the hard-constraint prefilter
    # (regulars.assembly.prefilter.plausible_pairs); an unscored pair was never plausible, or was
    # cut off by the per-user cap, and must stay ungroupable either way.
    plausible = frozenset(key for key, _ in scores.pairs())
    plans = assemble_round(
        user_ids,
        scores,
        plausible,
        previous_groups=(),
        rng=rng,
        exploration_rate=DEFAULT_EXPLORATION_RATE,
        group_size=GROUP_SIZE,
    )
    groups: list[repo.Group] = []
    for plan in plans:
        group = repo.create_group(conn, zip_code=zip_code, assembly_method=plan.assembly_method)
        for user_id in plan.members:
            repo.add_membership(conn, group.id, user_id, joined_round=1)
        for record in plan.audit:
            repo.write_audit(
                conn,
                actor="seed",
                decision_type=record.decision_type,
                reason=record.reason,
                payload=record.payload,
                subject_user_id=record.subject_user_id,
                group_id=group.id,
            )
        groups.append(group)
    return groups


def _simulate_round(
    engine: Engine,
    rng: random.Random,
    group: repo.Group,
    all_user_ids: list[str],
    scores: PairScores,
) -> int:
    """Simulate one completed dinner for ``group`` and call the real ``post_dinner``.

    Returns the number of members removed. One member is a paid no-show; another gets a clearly
    low approval rating from everyone else who attended, so ``prune_and_backfill`` has both
    removal rules to exercise.
    """
    with engine.begin() as conn:
        members = [m.user_id for m in repo.memberships_for_group(conn, group.id, active_only=True)]
        if len(members) < 3:
            return 0
        plan = repo.create_plan(conn, group.id)
        starts_at = datetime.now(UTC) - timedelta(days=2)
        repo.lock_plan(conn, plan.id, venue=rng.choice(_VENUE_POOL), starts_at=starts_at)

        no_show, low_rated, *rest = members
        attendees = [low_rated, *rest]

        for user_id in members:
            repo.upsert_rsvp(
                conn,
                plan.id,
                user_id,
                state="paid",
                deposit_intent_id=f"seed-dep-{plan.id}-{user_id}",
                paid_at=starts_at - timedelta(days=1),
            )
        for user_id in members:
            repo.record_attendance(
                conn, plan.id, user_id, attended=user_id != no_show, source="seed"
            )
        for rater in attendees:
            for ratee in attendees:
                if rater == ratee:
                    continue
                if ratee == low_rated:
                    would_see_again = False
                    felt_connected = rng.randint(1, 2)
                else:
                    would_see_again = rng.random() < 0.8
                    felt_connected = rng.randint(3, 7)
                repo.insert_rating(
                    conn,
                    plan.id,
                    rater_id=rater,
                    ratee_id=ratee,
                    would_see_again=would_see_again,
                    felt_connected_1_7=felt_connected,
                )
        plan_id = plan.id

    queue = sorted(set(all_user_ids) - set(members))[:8]
    update = post_dinner(engine, plan_id, queue=queue, scores=scores, min_keep=3)
    return len(update.removed)


def _is_seeded(engine: Engine) -> bool:
    with engine.connect() as conn:
        count = conn.execute(select(func.count()).select_from(schema.users)).scalar_one()
    return bool(count)


def _count(engine: Engine, table: Table) -> int:
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(table)).scalar_one())


def seed_database(
    engine: Engine, *, n_users: int = DEFAULT_USERS, zip_code: str = DEFAULT_ZIP, seed: int = 0
) -> SeedSummary:
    """Populate ``engine`` (schema already created) with a full, internally-consistent demo
    dataset. See the module docstring for the five steps."""
    schema.metadata.create_all(engine)
    rng = random.Random(seed)

    with engine.begin() as conn:
        users, personas = _seed_users_and_personas(conn, rng, seed, n_users, zip_code)

    with engine.begin() as conn:
        scores = _seed_sims_and_predictions(conn, rng, seed, users, personas)

    with engine.begin() as conn:
        groups = _open_groups(conn, rng, zip_code, users, scores)

    all_user_ids = [u.id for u in users]
    n_removed = 0
    n_rounds = 0
    for group in groups[:ROUNDS_TO_SIMULATE]:
        n_removed += _simulate_round(engine, rng, group, all_user_ids, scores)
        n_rounds += 1

    return SeedSummary(
        db_url=str(engine.url),
        n_users=_count(engine, schema.users),
        n_personas=_count(engine, schema.persona_versions),
        n_predictions=_count(engine, schema.predictions),
        n_groups=_count(engine, schema.groups),
        n_rounds_simulated=n_rounds,
        n_removed=n_removed,
        n_audit_rows=_count(engine, schema.audit_log),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m regulars.ops.seed")
    parser.add_argument("--users", type=int, default=DEFAULT_USERS, help="number of users to seed")
    parser.add_argument(
        "--zip", dest="zip_code", default=DEFAULT_ZIP, help="single launch zip code"
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible fake data")
    parser.add_argument(
        "--db-url",
        dest="db_url",
        default=None,
        help=f"SQLAlchemy URL; defaults to a fresh sqlite file at ./{DEFAULT_DB_PATH}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="seed even if the target database already has users in it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_url = args.db_url or f"sqlite:///{DEFAULT_DB_PATH}"
    engine = make_engine(db_url)
    schema.metadata.create_all(engine)

    if _is_seeded(engine) and not args.force:
        print(
            f"refusing to seed {engine.url}: it already has users. Pass --force to seed anyway, "
            f"or delete the file (default: {DEFAULT_DB_PATH}) to start fresh.",
            file=sys.stderr,
        )
        return 1

    summary = seed_database(engine, n_users=args.users, zip_code=args.zip_code, seed=args.seed)
    print(summary.render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
