"""Two-stage candidate ranking (§8 cost model).

Stage 1: every candidate gets one scenario × one seed on the cheap (Haiku-class) model.
Stage 2: the top ``first_pass_top_k`` get the full scenarios × seeds grid on the frontier model.

Scoring is injected (``score_fn``) so this module does not depend on ``regulars.judge``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from regulars.core.llm import LLMClient
from regulars.core.persona import PersonaSpec
from regulars.core.transcript import Transcript
from regulars.sim.cache import SimCache
from regulars.sim.harness import SimConfig, run_pair, single_shot
from regulars.sim.scenarios import COLD_OPEN, Scenario

ScoreFn = Callable[[list[Transcript]], float]


@dataclass(frozen=True)
class RankedCandidate:
    candidate_id: str
    cheap_score: float
    frontier_score: float | None  # None when the candidate did not make the top-k
    transcripts: tuple[Transcript, ...]  # frontier transcripts if run, else the cheap one

    @property
    def score(self) -> float:
        return self.frontier_score if self.frontier_score is not None else self.cheap_score


async def rank_candidates(
    user_spec: PersonaSpec,
    candidates: Mapping[str, PersonaSpec],
    cheap_llm: LLMClient,
    frontier_llm: LLMClient,
    score_fn: ScoreFn,
    first_pass_top_k: int = 40,
    *,
    config: SimConfig | None = None,
    first_pass_scenario: Scenario = COLD_OPEN,
    scenarios: Sequence[Scenario] | None = None,
    cache: SimCache | None = None,
    user_id: str = "user",
) -> list[RankedCandidate]:
    """Rank ``candidates`` against ``user_spec``. Top-k are ordered by frontier score, the rest by
    cheap score, both descending. ``first_pass_top_k <= 0`` disables the frontier pass."""
    if not candidates:
        return []
    config = config or SimConfig()
    sem = asyncio.Semaphore(config.max_concurrency)
    cheap_config = single_shot(config)

    async def cheap_pass(cid: str, spec: PersonaSpec) -> tuple[str, float, list[Transcript]]:
        transcripts = await run_pair(
            cheap_llm,
            cheap_llm,
            user_spec,
            spec,
            cheap_config,
            scenarios=[first_pass_scenario],
            cache=cache,
            participants=(user_id, cid),
            semaphore=sem,
        )
        return cid, score_fn(transcripts), transcripts

    cheap = await asyncio.gather(*(cheap_pass(cid, s) for cid, s in candidates.items()))
    cheap_sorted = sorted(cheap, key=lambda r: r[1], reverse=True)
    top = cheap_sorted[: max(first_pass_top_k, 0)]
    rest = cheap_sorted[max(first_pass_top_k, 0) :]

    async def frontier_pass(cid: str, cheap_score: float) -> RankedCandidate:
        transcripts = await run_pair(
            frontier_llm,
            frontier_llm,
            user_spec,
            candidates[cid],
            config,
            scenarios=scenarios,
            cache=cache,
            participants=(user_id, cid),
            semaphore=sem,
        )
        return RankedCandidate(cid, cheap_score, score_fn(transcripts), tuple(transcripts))

    ranked = list(await asyncio.gather(*(frontier_pass(cid, sc) for cid, sc, _ in top)))
    ranked.sort(key=lambda r: r.score, reverse=True)
    ranked += [RankedCandidate(cid, sc, None, tuple(ts)) for cid, sc, ts in rest]
    return ranked
