"""Persona × persona simulation harness (§3).

``run_sim`` plays one scene: two ``LLMClient``s alternate turns, each seeing the dialogue so far
from its own point of view (its lines as ``assistant``, the counterpart's as ``user``), under an
anti-sycophancy system prompt (``regulars.sim.prompts``). ``run_pair`` runs the full
scenarios × seeds grid for a pair, consulting a ``SimCache`` first.

Friction injection: for exactly one seed per pair (``config.seeds[config.friction_seed_index]``)
each persona's ``friction_points`` are injected into the *counterpart's* scene, so at least one
transcript per pair contains the things each side dislikes actually happening.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace

from regulars.core.llm import LLMClient, LLMMessage
from regulars.core.persona import PersonaSpec
from regulars.core.transcript import Transcript, Turn
from regulars.sim.cache import SimCache, SimCacheKey
from regulars.sim.prompts import enactment_system_prompt
from regulars.sim.scenarios import SCENARIOS, Scenario

PERSONA_HASH_LEN = 12

# The Messages API requires the first message to be ``user``; a persona that opens the scene has
# no counterpart line to respond to, so it gets this neutral cue instead.
OPENING_CUE = "(You are the first to speak. The scene has started.)"


@dataclass(frozen=True)
class SimConfig:
    turns: int = 40
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    temperature: float = 0.95  # §3: 0.9-1.0, never 0.3
    friction_seed_index: int = 0
    max_tokens: int = 160  # ≤3 short sentences
    max_concurrency: int = 8

    @property
    def friction_seed(self) -> int | None:
        if not self.seeds:
            return None
        return self.seeds[self.friction_seed_index % len(self.seeds)]


def persona_hash(spec: PersonaSpec) -> str:
    """First 12 hex chars of sha256 over the canonical (sorted-key) JSON of the spec."""
    canonical = json.dumps(spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:PERSONA_HASH_LEN]


def pair_model_name(llm_a: LLMClient, llm_b: LLMClient) -> str:
    return llm_a.model if llm_a.model == llm_b.model else f"{llm_a.model}+{llm_b.model}"


def _history_for(speaker: str, turns: Sequence[Turn]) -> list[LLMMessage]:
    messages = [
        LLMMessage(role="assistant" if t.speaker_id == speaker else "user", content=t.text)
        for t in turns
    ]
    if not messages or messages[0].role != "user":
        messages.insert(0, LLMMessage(role="user", content=OPENING_CUE))
    return messages


def _clean_line(text: str, speaker: str) -> str:
    line = text.strip()
    prefix = f"{speaker}:"
    if line.lower().startswith(prefix.lower()):
        line = line[len(prefix) :].strip()
    return line or "..."


async def run_sim(
    llm_a: LLMClient,
    llm_b: LLMClient,
    spec_a: PersonaSpec,
    spec_b: PersonaSpec,
    scenario: Scenario,
    seed: int,
    config: SimConfig | None = None,
    *,
    participants: tuple[str, str] = ("a", "b"),
) -> Transcript:
    """One scene, ``config.turns`` generated turns. A opens on even seeds, B on odd."""
    config = config or SimConfig()
    id_a, id_b = participants
    scenario = scenario.for_pair(spec_a, spec_b)
    inject = seed == config.friction_seed
    systems = {
        id_a: enactment_system_prompt(
            spec_a, scenario, id_b, spec_b.social.friction_points if inject else ()
        ),
        id_b: enactment_system_prompt(
            spec_b, scenario, id_a, spec_a.social.friction_points if inject else ()
        ),
    }
    llms: dict[str, LLMClient] = {id_a: llm_a, id_b: llm_b}
    order = (id_a, id_b) if seed % 2 == 0 else (id_b, id_a)

    turns: list[Turn] = [
        Turn(speaker_id=order[i % 2], text=line) for i, line in enumerate(scenario.seed_turns)
    ]
    for _ in range(config.turns):
        speaker = order[len(turns) % 2]
        reply = await llms[speaker].complete(
            systems[speaker],
            _history_for(speaker, turns),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            seed=seed,
        )
        turns.append(Turn(speaker_id=speaker, text=_clean_line(reply, speaker)))

    return Transcript(
        participants=[id_a, id_b],
        turns=turns,
        source="sim",
        metadata={
            "scenario_id": scenario.id,
            "seed": seed,
            "persona_hash_a": persona_hash(spec_a),
            "persona_hash_b": persona_hash(spec_b),
            "model_a": llm_a.model,
            "model_b": llm_b.model,
            "temperature": config.temperature,
            "friction_injected": int(inject),
            "seed_turns": len(scenario.seed_turns),
            "opener": order[0],
        },
    )


async def run_pair(
    llm_a: LLMClient,
    llm_b: LLMClient,
    spec_a: PersonaSpec,
    spec_b: PersonaSpec,
    config: SimConfig | None = None,
    *,
    scenarios: Sequence[Scenario] | None = None,
    cache: SimCache | None = None,
    participants: tuple[str, str] = ("a", "b"),
    semaphore: asyncio.Semaphore | None = None,
) -> list[Transcript]:
    """scenarios × seeds for one pair, at most ``config.max_concurrency`` sims in flight.

    Results are ordered scenario-major, seed-minor regardless of completion order. Pass a shared
    ``semaphore`` to bound concurrency across many pairs.
    """
    config = config or SimConfig()
    scenes = list(SCENARIOS.values()) if scenarios is None else list(scenarios)
    sem = semaphore or asyncio.Semaphore(config.max_concurrency)
    hash_a, hash_b = persona_hash(spec_a), persona_hash(spec_b)
    model = pair_model_name(llm_a, llm_b)

    async def one(scenario: Scenario, seed: int) -> Transcript:
        key = SimCacheKey(hash_a, hash_b, scenario.id, seed, model)
        if cache is not None:
            hit = cache.get(key)
            if hit is not None:
                return hit
        async with sem:
            transcript = await run_sim(
                llm_a, llm_b, spec_a, spec_b, scenario, seed, config, participants=participants
            )
        if cache is not None:
            cache.put(key, transcript)
        return transcript

    return list(await asyncio.gather(*(one(sc, seed) for sc in scenes for seed in config.seeds)))


def single_shot(config: SimConfig, seed: int | None = None) -> SimConfig:
    """The cheap first-pass config: one seed (the friction seed by default, so the one sim a
    pair gets is the hardest one), everything else unchanged."""
    chosen = config.friction_seed if seed is None else seed
    if chosen is None:
        raise ValueError("SimConfig has no seeds")
    return replace(config, seeds=(chosen,), friction_seed_index=0)
