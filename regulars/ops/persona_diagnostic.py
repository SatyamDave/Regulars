"""§6 Problem 4, run on real people: "watch variance before mean," before you have a single group.

Interview 5-10 real people (three typed or transcribed answers each, per
``regulars.persona.interview.INTERVIEW_PROMPTS``), build personas, simulate every pair, and check
whether the simulation discriminates at all. This is the build-order step 4 gate
(technical-architecture.md §11: "sim harness + variance diagnostic ... σ(y_hat) non-degenerate,
AUC > 0.6 on dev set") run against real personas instead of unit-test fixtures. If every pair
scores the same, the sim is agreeable mush and nothing downstream is worth trusting — find that
out on 10 friends, not on a filled restaurant.

Usage::

    python -m regulars.ops.persona_diagnostic template --out people.yaml
    # ... fill in real answers, optionally a few known would_see_again labels ...
    python -m regulars.ops.persona_diagnostic personas --input people.yaml --out personas/
    python -m regulars.ops.persona_diagnostic variance --personas personas/ --input people.yaml

Both ``personas`` and ``variance`` call a real LLM via ``regulars.llm.factory.llm_from_env`` (set
``ANTHROPIC_API_KEY``, or point ``REGULARS_ENACTMENT_BASE_URL`` at Centaur). ``variance`` runs one
short simulated conversation per pair by default (a single scenario, a handful of turns)
specifically to keep the first check cheap; pass ``--full`` for the real §3 grid (3 scenarios x 5
seeds) once the light pass looks discriminative.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import yaml
from pydantic import BaseModel, Field

from regulars.core import LLMClient, PairFeatures, PersonaSpec, Transcript
from regulars.features.extract import ExtractorConfig, extract_pair_features
from regulars.judge.diagnostics import auc_on_known_pairs
from regulars.judge.v1 import DEFAULT_VERSION, JudgeV1, pair_score
from regulars.llm.factory import llm_from_env
from regulars.persona.extract import extract_persona
from regulars.persona.interview import INTERVIEW_PROMPTS, InterviewAnswer
from regulars.sim.cache import InMemorySimCache, SimCache
from regulars.sim.harness import SimConfig, run_pair
from regulars.sim.scenarios import COLD_OPEN, SCENARIOS, Scenario
from regulars.sim.variance import VarianceReport, sim_variance_report

TEMPLATE = """\
# Ten people you know well enough to already know who'd click and who wouldn't. Real answers to
# the three interview prompts (regulars/bot/voice.py INTERVIEW_PROMPTS) -- typed is fine, this
# doesn't need real voice memos to test the variance diagnostic.
people:
  - id: priya
    answers:
      - "Walk me through your last genuinely good weekend answer for priya..."
      - "Someone I stopped hanging out with, and why, for priya..."
      - "What I'm actually looking for right now, for priya..."
  - id: sam
    answers:
      - "..."
      - "..."
      - "..."

# Optional. A handful of pairs where you ALREADY know the answer, so `variance` also reports an
# AUC (technical-architecture.md §11 build-order step 1's gate, and §6's "AUC < 0.6 means the
# harness is broken" check). Leave empty if you'd rather run this purely as a variance check.
labels:
  - {a: priya, b: sam, would_see_again: true}
"""


class Person(BaseModel):
    id: str
    answers: list[str] = Field(min_length=3, max_length=3)


class Label(BaseModel):
    a: str
    b: str
    would_see_again: bool


class PeopleFile(BaseModel):
    people: list[Person]
    labels: list[Label] = Field(default_factory=list)


def load_people_file(path: Path) -> PeopleFile:
    return PeopleFile.model_validate(yaml.safe_load(path.read_text()))


def _pair_key(a: str, b: str) -> str:
    lo, hi = sorted((a, b))
    return f"{lo}|{hi}"


async def build_personas(
    people: Sequence[Person], llm: LLMClient, *, concurrency: int = 3
) -> dict[str, PersonaSpec]:
    """Real ``extract_persona`` call per person; §2's negative-space re-ask runs for real too."""
    sem = asyncio.Semaphore(concurrency)

    async def one(person: Person) -> tuple[str, PersonaSpec]:
        answers = [
            InterviewAnswer(prompt_index=i, text=text, typed=True)
            for i, text in enumerate(person.answers)
        ]
        async with sem:
            spec = await extract_persona(llm, answers, prosody=None)
        return person.id, spec

    results = await asyncio.gather(*(one(p) for p in people))
    return dict(results)


def write_personas(personas: dict[str, PersonaSpec], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for person_id, spec in personas.items():
        (out_dir / f"{person_id}.json").write_text(spec.model_dump_json(indent=2))


def load_personas(personas_dir: Path) -> dict[str, PersonaSpec]:
    return {
        p.stem: PersonaSpec.model_validate_json(p.read_text())
        for p in sorted(personas_dir.glob("*.json"))
    }


@dataclass(frozen=True)
class VarianceRunResult:
    report: VarianceReport
    ranked_pairs: list[tuple[str, str, float]]  # (a, b, mean score), descending
    auc: float | None
    auc_note: str | None


async def run_variance_diagnostic(
    personas: dict[str, PersonaSpec],
    llm: LLMClient,
    *,
    config: SimConfig,
    scenarios: Sequence[Scenario],
    labels: Sequence[Label] = (),
    cache: SimCache | None = None,
    concurrency: int = 4,
) -> VarianceRunResult:
    """Simulate every unordered pair, score with a judge fit on this population, report variance.

    One sim (or the full §3 grid, via ``config``/``scenarios``) per pair, scored with the
    heuristic feature extractor (free, no extra LLM calls beyond the simulation itself) and a
    ``JudgeV1`` fit fresh on this population's own features -- this diagnostic asks "does the sim
    discriminate at all," not "does it match a pre-tuned judge."
    """
    ids = sorted(personas)
    pairs = list(itertools.combinations(ids, 2))
    sim_cache = cache or InMemorySimCache()
    sem = asyncio.Semaphore(concurrency)
    fx_config = ExtractorConfig.heuristic()

    async def simulate(a: str, b: str) -> tuple[tuple[str, str], list[Transcript]]:
        async with sem:
            transcripts = await run_pair(
                llm,
                llm,
                personas[a],
                personas[b],
                config,
                scenarios=scenarios,
                cache=sim_cache,
                participants=(a, b),
            )
        return (a, b), transcripts

    all_transcripts: dict[tuple[str, str], list[Transcript]] = dict(
        await asyncio.gather(*(simulate(a, b) for a, b in pairs))
    )

    population: list[PairFeatures] = []
    features_by_pair: dict[tuple[str, str], list[tuple[PairFeatures, PairFeatures]]] = {}
    for (a, b), transcripts in all_transcripts.items():
        directed: list[tuple[PairFeatures, PairFeatures]] = []
        for t in transcripts:
            pf_ab, pf_ba = await extract_pair_features(t, a, b, fx_config)
            population.extend([pf_ab, pf_ba])
            directed.append((pf_ab, pf_ba))
        features_by_pair[a, b] = directed

    # A judge fit fresh on THIS population, not DEFAULT_WEIGHTS pinned by an earlier run --
    # this diagnostic asks whether the sim discriminates at all, before any weight is trusted.
    judge = JudgeV1.default(population, version=f"{DEFAULT_VERSION}-diagnostic")

    scores_by_pair: dict[str, list[float]] = {}
    means: list[tuple[str, str, float]] = []
    for (a, b), directed in features_by_pair.items():
        pair_scores = [
            pair_score(judge.score(pf_ab), judge.score(pf_ba)) for pf_ab, pf_ba in directed
        ]
        scores_by_pair[_pair_key(a, b)] = pair_scores
        means.append((a, b, fmean(pair_scores)))

    report = sim_variance_report(scores_by_pair)
    ranked = sorted(means, key=lambda row: row[2], reverse=True)

    auc: float | None = None
    auc_note: str | None = None
    if labels:
        y_hat_by_pair = {(a, b): score for a, b, score in means}
        y_hat_by_pair.update({(b, a): score for a, b, score in means})
        ys, labs = [], []
        for lab in labels:
            key = (lab.a, lab.b) if (lab.a, lab.b) in y_hat_by_pair else (lab.b, lab.a)
            if key not in y_hat_by_pair:
                continue
            ys.append(y_hat_by_pair[key])
            labs.append(1 if lab.would_see_again else 0)
        if len(set(labs)) < 2:
            auc_note = "labels given, but all one class (need at least one yes and one no)"
        elif len(ys) < 2:
            auc_note = "no labeled pair matched a simulated pair"
        else:
            auc = auc_on_known_pairs(ys, labs)

    return VarianceRunResult(report=report, ranked_pairs=ranked, auc=auc, auc_note=auc_note)


def render_report(result: VarianceRunResult) -> str:
    r = result.report
    lines = [
        "# Persona variance diagnostic",
        "",
        f"pairs simulated:        {r.n_pairs}",
        f"between-pair std:       {r.between_pair_std:.4f}",
        f"mean within-pair std:   {r.mean_within_pair_std:.4f}",
        f"degenerate:             {r.degenerate}",
    ]
    if r.reason:
        lines.append(f"reason:                 {r.reason}")
    if result.auc is not None:
        lines.append(
            f"AUC vs your labels:     {result.auc:.3f}  (< 0.6 means the harness is broken)"
        )
    elif result.auc_note:
        lines.append(f"AUC:                    not computed ({result.auc_note})")
    lines += ["", "## Verdict", ""]
    if r.degenerate:
        lines.append(
            "**FAIL.** The sim does not discriminate between these pairs. Every score is "
            "noise; do not trust a single number out of this pipeline until this passes. "
            "Re-check: are `friction_points`/`drains` actually populated per persona (§2)? Is "
            "temperature at 0.9-1.0? Try `--full` for the real 3-scenario x 5-seed grid before "
            "concluding the harness itself is broken."
        )
    else:
        lines.append(
            "**PASS.** Between-pair spread clears the seed-to-seed noise floor -- the sim is "
            "discriminating between at least some of these pairs. This does not yet mean the "
            "*ranking* is right (that's the known-pairs AUC check, `regulars.ops.known_pairs` / "
            "`regulars.judge.cli`) -- only that it isn't uniformly agreeable."
        )
    lines += ["", "## Pairs, highest score first", ""]
    for a, b, score in result.ranked_pairs:
        lines.append(f"{score:+.3f}  {a} <-> {b}")
    return "\n".join(lines) + "\n"


def _scenario_list(names: str | None) -> list[Scenario]:
    if not names:
        return [COLD_OPEN]
    return [SCENARIOS[n.strip()] for n in names.split(",")]


def cmd_template(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} already exists; pass --force to overwrite", file=sys.stderr)
        return 1
    out.write_text(TEMPLATE)
    print(f"wrote {out}")
    print(f"(the three prompts are: {INTERVIEW_PROMPTS})")
    return 0


def cmd_personas(args: argparse.Namespace) -> int:
    people_file = load_people_file(Path(args.input))
    llm = llm_from_env(args.llm)
    personas = asyncio.run(build_personas(people_file.people, llm))
    write_personas(personas, Path(args.out))
    missing_negative_space = [
        pid for pid, spec in personas.items() if not spec.has_negative_space()
    ]
    print(f"wrote {len(personas)} personas to {args.out}")
    if missing_negative_space:
        print(
            f"WARNING: no friction_points/drains for: {', '.join(missing_negative_space)} "
            "-- §2 says this is load-bearing for sim variance; consider a richer interview answer",
            file=sys.stderr,
        )
    return 0


def cmd_variance(args: argparse.Namespace) -> int:
    personas = load_personas(Path(args.personas))
    if len(personas) < 2:
        print(
            f"need at least 2 personas in {args.personas}, found {len(personas)}", file=sys.stderr
        )
        return 1
    people_file = load_people_file(Path(args.input)) if args.input else PeopleFile(people=[])
    llm = llm_from_env(args.llm)
    n_pairs = len(personas) * (len(personas) - 1) // 2
    config = SimConfig(turns=args.turns, seeds=tuple(range(args.seeds)))
    scenarios = list(SCENARIOS.values()) if args.full else _scenario_list(args.scenarios)
    n_sims = n_pairs * len(scenarios) * len(config.seeds)
    print(f"{len(personas)} personas -> {n_pairs} pairs -> {n_sims} simulated conversations")
    if not args.yes:
        reply = input("proceed? [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("aborted")
            return 1
    result = asyncio.run(
        run_variance_diagnostic(
            personas, llm, config=config, scenarios=scenarios, labels=people_file.labels
        )
    )
    text = render_report(result)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"(also wrote {args.out})")
    return 1 if result.report.degenerate else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m regulars.ops.persona_diagnostic")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("template", help="write an example people.yaml")
    t.add_argument("--out", default="people.yaml")
    t.add_argument("--force", action="store_true")
    t.set_defaults(func=cmd_template)

    pp = sub.add_parser("personas", help="build a PersonaSpec per person via a real LLM call")
    pp.add_argument("--input", required=True)
    pp.add_argument("--out", required=True)
    pp.add_argument("--llm", default="scoring", choices=["scoring", "cheap", "enactment"])
    pp.set_defaults(func=cmd_personas)

    v = sub.add_parser("variance", help="simulate every pair and check sigma(y_hat)")
    v.add_argument("--personas", required=True)
    v.add_argument("--input", help="the people.yaml with optional labels, for an AUC check too")
    v.add_argument("--llm", default="enactment", choices=["scoring", "cheap", "enactment"])
    v.add_argument(
        "--turns", type=int, default=16, help="short by default to keep the first pass cheap"
    )
    v.add_argument("--seeds", type=int, default=2)
    v.add_argument("--scenarios", default="cold_open", help="comma-separated scenario ids")
    v.add_argument(
        "--full", action="store_true", help="the real §3 grid: all 3 scenarios x 5 seeds"
    )
    v.add_argument("--out", help="also write the report to this path")
    v.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    v.set_defaults(func=cmd_variance)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
