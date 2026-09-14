"""Hard-constraint prefilter (technical-architecture.md §7.1, §7.4, §8).

Everything here is a *filter*. A pair that fails a hard constraint is simply absent from the
result; nothing downstream ever sees it, so it can never be traded off against a good score
(§7.4: hard constraints as filters before the optimizer, never as score penalties).

Constraints, all of which must hold for a pair to be plausible:

- **Zip**: same zip, or adjacent under the supplied ``zip_adjacency`` (symmetrised).
- **Availability**: at least one shared slot.
- **Exclusions**: neither user's stated exclusions intersect the other's tags.

The per-user cap (``max_per_user``, default 200 per §8) bounds the number of simulations the
judge has to run. It is applied deterministically for a given ``seed``: plausible pairs are
shuffled with ``random.Random(seed)`` and accepted greedily while both endpoints are under the
cap, so the same inputs always yield the same candidate set and no user ever exceeds the cap.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass

Pair = tuple[str, str]
"""An undirected pair, always stored with ``a < b``. Build one with :func:`pair_key`."""

ZipAdjacency = Mapping[str, Collection[str]]

RejectionReason = str


def pair_key(a: str, b: str) -> Pair:
    if a == b:
        raise ValueError(f"a pair needs two distinct users, got {a!r} twice")
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True)
class Candidate:
    user_id: str
    zip: str
    availability: frozenset[str] = frozenset()
    exclusions: frozenset[str] = frozenset()
    tags: frozenset[str] = frozenset()


def zips_compatible(zip_a: str, zip_b: str, zip_adjacency: ZipAdjacency | None = None) -> bool:
    if zip_a == zip_b:
        return True
    if zip_adjacency is None:
        return False
    return zip_b in zip_adjacency.get(zip_a, ()) or zip_a in zip_adjacency.get(zip_b, ())


def rejection_reason(
    a: Candidate, b: Candidate, zip_adjacency: ZipAdjacency | None = None
) -> RejectionReason | None:
    """``None`` when the pair is plausible, otherwise the first hard constraint it fails."""
    if not zips_compatible(a.zip, b.zip, zip_adjacency):
        return "zip"
    if not (a.availability & b.availability):
        return "availability"
    if (a.exclusions & b.tags) or (b.exclusions & a.tags):
        return "exclusion"
    return None


def is_plausible(a: Candidate, b: Candidate, zip_adjacency: ZipAdjacency | None = None) -> bool:
    return rejection_reason(a, b, zip_adjacency) is None


def plausible_pairs(
    candidates: Iterable[Candidate],
    max_per_user: int = 200,
    zip_adjacency: ZipAdjacency | None = None,
    seed: int = 0,
) -> frozenset[Pair]:
    """All pairs passing every hard constraint, capped at ``max_per_user`` partners per user.

    O(n²) in candidates but with no simulation cost; it exists precisely so the quadratic step is
    this cheap one and not the judge (§8).
    """
    if max_per_user < 1:
        raise ValueError("max_per_user must be at least 1")
    by_id: dict[str, Candidate] = {}
    for c in candidates:
        if c.user_id in by_id:
            raise ValueError(f"duplicate candidate user_id {c.user_id!r}")
        by_id[c.user_id] = c
    ordered = [by_id[u] for u in sorted(by_id)]

    pairs: list[Pair] = [
        pair_key(a.user_id, b.user_id)
        for i, a in enumerate(ordered)
        for b in ordered[i + 1 :]
        if is_plausible(a, b, zip_adjacency)
    ]
    random.Random(seed).shuffle(pairs)

    degree: Counter[str] = Counter()
    kept: list[Pair] = []
    for a_id, b_id in pairs:
        if degree[a_id] < max_per_user and degree[b_id] < max_per_user:
            kept.append((a_id, b_id))
            degree[a_id] += 1
            degree[b_id] += 1
    return frozenset(kept)
