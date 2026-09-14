"""The randomized exploration arm (technical-architecture.md §6 Problem 1).

Roughly ``exploration_rate`` (30%) of tables are assembled at random and tagged
``assembly_method='random'``. Their ratings are the only unbiased evaluation set: for
model-assembled tables the outcome is observed only on pairs the model chose, so scoring the
model on them is contaminated.

The random arm differs from the model arm in **scoring only**. Both arms take the same users,
the same ``group_size`` and the same prefiltered ``plausible`` set (hard constraints are hard
in both arms, §7.4). The model arm optimises the minimum pair score; the random arm ignores
scores entirely and samples a partition uniformly by rejection: shuffle, chunk, accept if every
within-group pair is plausible.

Arm assignment happens at the *user* level before either optimizer runs: ``assemble_round``
draws a method per table slot, then randomly carves off ``group_size`` users per random slot.
Users therefore land in the random arm independently of their scores, which is what makes the
arm's labels unbiased. Randomly-armed users the random arm cannot seat (see ``relaxed``) fall
through to the model arm rather than being dropped.
"""

from __future__ import annotations

import random
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Literal

from regulars.assembly.audit import AuditRecord, json_score, members_key
from regulars.assembly.optimizer import (
    DEFAULT_CONTINUITY_BONUS,
    Continuity,
    PairScores,
    assemble_groups,
    group_score,
)
from regulars.assembly.prefilter import Pair, pair_key

AssemblyMethod = Literal["model", "random"]

DEFAULT_EXPLORATION_RATE = 0.30
DEFAULT_MAX_TRIES = 1_000


def choose_method(
    rng: random.Random, exploration_rate: float = DEFAULT_EXPLORATION_RATE
) -> AssemblyMethod:
    """Bernoulli draw: ``'random'`` with probability ``exploration_rate``."""
    if not 0.0 <= exploration_rate <= 1.0:
        raise ValueError("exploration_rate must be in [0, 1]")
    return "random" if rng.random() < exploration_rate else "model"


@dataclass(frozen=True)
class RandomGroups:
    groups: tuple[tuple[str, ...], ...]
    leftovers: tuple[str, ...]
    relaxed: bool
    """True when uniform rejection sampling failed and the greedy fallback was used."""
    tries: int
    audit: tuple[AuditRecord, ...]


def _all_plausible(group: Collection[str], plausible: Collection[Pair]) -> bool:
    ms = list(group)
    return all(pair_key(a, b) in plausible for i, a in enumerate(ms) for b in ms[i + 1 :])


def random_groups(
    users: Iterable[str],
    group_size: int,
    plausible: Collection[Pair],
    rng: random.Random,
    max_tries: int = DEFAULT_MAX_TRIES,
) -> RandomGroups:
    """Random partition into groups of ``group_size`` in which every pair is plausible.

    Rejection sampling for up to ``max_tries`` shuffles (uniform over valid partitions). If none
    is found the constraint that is relaxed is *seating everyone*, never plausibility: a
    randomised greedy placement seats whoever it can and leaves the rest as leftovers, and the
    result is flagged ``relaxed=True`` because that sample is no longer uniform.
    """
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    if max_tries < 1:
        raise ValueError("max_tries must be at least 1")
    pool = sorted(set(users))
    n_groups = len(pool) // group_size
    if n_groups == 0:
        return RandomGroups((), tuple(pool), False, 0, ())

    tries = 0
    groups: list[list[str]] = []
    relaxed = False
    for _ in range(max_tries):
        tries += 1
        rng.shuffle(pool)
        chunks = [pool[i * group_size : (i + 1) * group_size] for i in range(n_groups)]
        if all(_all_plausible(c, plausible) for c in chunks):
            groups = chunks
            break
    else:
        relaxed = True
        rng.shuffle(pool)
        open_groups: list[list[str]] = []
        for u in pool:
            for g in open_groups:
                if len(g) < group_size and all(pair_key(u, m) in plausible for m in g):
                    g.append(u)
                    break
            else:
                if len(open_groups) < n_groups:
                    open_groups.append([u])
        groups = [g for g in open_groups if len(g) == group_size]

    seated = {u for g in groups for u in g}
    final = tuple(members_key(g) for g in groups)
    reason = (
        "This table was drawn at random from people who all cleared the hard filters, so what "
        "happens here tells us how well the model does compared with chance."
    )
    audit = tuple(
        AuditRecord(
            decision_type="group_assembled",
            subject_user_id=None,
            group_members=g,
            reason=reason,
            payload={
                "assembly_method": "random",
                "group_size": group_size,
                "relaxed": relaxed,
                "tries": tries,
            },
        )
        for g in final
    )
    return RandomGroups(
        groups=final,
        leftovers=tuple(sorted(u for u in pool if u not in seated)),
        relaxed=relaxed,
        tries=tries,
        audit=audit,
    )


@dataclass(frozen=True)
class GroupPlan:
    members: tuple[str, ...]
    assembly_method: AssemblyMethod
    min_score: float
    """Raw minimum pair score (``-inf`` if a pair is unscored, possible only in the random arm)."""
    audit: tuple[AuditRecord, ...]


def _method_record(
    members: tuple[str, ...], method: AssemblyMethod, exploration_rate: float, min_score: float
) -> AuditRecord:
    if method == "random":
        reason = (
            f"Exploration arm: this table was assembled at random (probability "
            f"{exploration_rate:.0%}) so its ratings give an unbiased read on the model."
        )
    else:
        reason = (
            f"Model arm: this table was assembled by the model (probability "
            f"{1 - exploration_rate:.0%} of tables are)."
        )
    return AuditRecord(
        decision_type="method_chosen",
        subject_user_id=None,
        group_members=members,
        reason=reason,
        payload={
            "assembly_method": method,
            "exploration_rate": exploration_rate,
            "min_score": json_score(min_score),
        },
    )


def assemble_round(
    users: Iterable[str],
    scores: PairScores,
    plausible: Collection[Pair],
    previous_groups: Iterable[Collection[str]],
    rng: random.Random,
    exploration_rate: float = DEFAULT_EXPLORATION_RATE,
    *,
    group_size: int = 6,
    continuity_bonus: float = DEFAULT_CONTINUITY_BONUS,
    iterations: int = 200,
) -> list[GroupPlan]:
    """One round of assembly with the exploration arm; ``assembly_method`` set on every group.

    Users not in any returned plan are this round's leftovers (see :func:`leftover_users`).
    Scores are masked to ``plausible`` before the model arm sees them, so an implausible pair is
    unscored (``-inf``) and can never co-group in either arm. The continuity bonus applies to the
    model arm only.
    """
    pool = sorted(set(users))
    rng.shuffle(pool)
    n_slots = len(pool) // group_size
    methods = [choose_method(rng, exploration_rate) for _ in range(n_slots)]
    n_random = sum(1 for m in methods if m == "random")
    random_users = pool[: n_random * group_size]
    model_users = pool[n_random * group_size :]

    masked = scores.masked(plausible)
    rg = random_groups(random_users, group_size, plausible, rng)
    prev = list(previous_groups)
    continuity = Continuity(prev, continuity_bonus) if prev else None
    mr = assemble_groups(
        [*model_users, *rg.leftovers],
        masked,
        group_size=group_size,
        continuity=continuity,
        iterations=iterations,
        seed=rng.randrange(2**31),
    )

    plans: list[GroupPlan] = []
    for members, record in zip(rg.groups, rg.audit, strict=True):
        m = group_score(members, masked)
        plans.append(
            GroupPlan(
                members=members,
                assembly_method="random",
                min_score=m,
                audit=(_method_record(members, "random", exploration_rate, m), record),
            )
        )
    for members, record, m in zip(mr.groups, mr.audit, mr.per_group_min, strict=True):
        plans.append(
            GroupPlan(
                members=members,
                assembly_method="model",
                min_score=m,
                audit=(_method_record(members, "model", exploration_rate, m), record),
            )
        )
    return plans


def leftover_users(users: Iterable[str], plans: Iterable[GroupPlan]) -> tuple[str, ...]:
    seated = {u for p in plans for u in p.members}
    return tuple(sorted(set(users) - seated))
