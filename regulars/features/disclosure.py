"""Self-disclosure depth and reciprocity (docs/technical-architecture.md §4).

Each spoken turn is scored 0..3:

    0  nothing about the self (or too short to tell)
    1  first-person factual ("I moved here in March")
    2  first-person with affect ("I was nervous about moving")
    3  affect plus intensity or dwelling on it ("honestly I was terrified for months, ...")

``disclosure_reciprocity`` is the lag-1 Pearson correlation between a's depth and b's
next-turn depth; ``disclosure_asymmetry`` is |mean depth a - mean depth b| (did one person carry
it). ``HeuristicDisclosureScorer`` is the lexicon first pass; ``LLMDisclosureScorer`` scores all
turns in ONE call.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from regulars.core.llm import LLMClient
from regulars.core.transcript import Transcript, Turn
from regulars.features._llm import batch_labels
from regulars.features._text import tokenize, words

Depth = int  # 0..3

FIRST_PERSON: frozenset[str] = words("i me my mine myself i'm i've i'd i'll we our us we're we've")
AFFECT: frozenset[str] = words(
    """
    feel feels felt feeling feelings scared scary afraid terrified anxious anxiety nervous lonely
    alone happy sad miss missed missing hurt hurts proud ashamed embarrassed embarrassing worried
    worry worries stressed stress cry cried crying angry mad furious grateful struggle struggled
    struggling struggles therapy therapist divorce divorced died death dying depressed depression
    grief grieving heartbroken love loved hate hated regret regrets guilty guilt insecure
    overwhelmed exhausted burnt burnout panic panicked relieved relief vulnerable trust trusted
    lost failing failed failure broke breakup dumped fired jealous resent resentment shame
    trauma sober sobriety addiction relapse pregnant miscarriage sick illness diagnosis
    """
)
INTENSIFIERS: frozenset[str] = words(
    """
    really so very honestly deeply truly totally completely genuinely seriously incredibly
    extremely absolutely desperately never always constantly literally
    """
)
_VULNERABLE_PHRASES = re.compile(
    r"\bto be honest\b|\bi(?:'ve| have) never told\b|"
    r"\bi don't (?:usually|normally) (?:tell|talk)\b|\bbetween us\b|"
    r"\bi(?:'m| am) not (?:proud|okay|ok|fine)\b|\bhardest (?:thing|part|year)\b"
)


class DisclosureScorer(Protocol):
    """One depth (0..3) per spoken turn (``transcript.spoken()`` order)."""

    async def score(self, transcript: Transcript) -> list[Depth]: ...


@dataclass(frozen=True)
class HeuristicDisclosureScorer:
    """First person + affect lexicon + intensifiers / length."""

    min_words: int = 3
    long_turn_words: int = 15

    def score_text(self, text: str) -> Depth:
        tokens = tokenize(text)
        if len(tokens) < self.min_words:
            return 0
        toks = set(tokens)
        if not toks & FIRST_PERSON:
            return 0
        lower = text.lower()
        affect = bool(toks & AFFECT) or bool(_VULNERABLE_PHRASES.search(lower))
        if not affect:
            return 1
        intense = bool(toks & INTENSIFIERS) or _VULNERABLE_PHRASES.search(lower) is not None
        if intense or len(tokens) >= self.long_turn_words:
            return 3
        return 2

    def score_turns(self, turns: Sequence[Turn]) -> list[Depth]:
        return [self.score_text(t.text) for t in turns]

    async def score(self, transcript: Transcript) -> list[Depth]:
        return self.score_turns(transcript.spoken())


_SYSTEM = """You rate the self-disclosure depth of each turn in a two-person conversation:
0 = nothing personal revealed; 1 = personal facts or preferences; 2 = personal feelings,
opinions held with some vulnerability, or relationships; 3 = deep disclosure — fears, shame,
grief, secrets, things the speaker rarely tells people. Answer with a JSON array of integers
only."""


@dataclass(frozen=True)
class LLMDisclosureScorer:
    """Scores every turn in one ``LLMClient`` call; invalid entries fall back to the heuristic."""

    client: LLMClient
    fallback: HeuristicDisclosureScorer = field(default_factory=HeuristicDisclosureScorer)

    async def score(self, transcript: Transcript) -> list[Depth]:
        turns = transcript.spoken()
        return await batch_labels(
            self.client,
            system=_SYSTEM,
            turns=turns,
            fallback=self.fallback.score_turns(turns),
            normalize=_normalize,
            per_turn_tokens=3,
        )


def _normalize(item: Any) -> Depth | None:
    if isinstance(item, bool):
        return None
    if isinstance(item, int | float):
        v = int(round(item))
    elif isinstance(item, str) and item.strip().isdigit():
        v = int(item.strip())
    else:
        return None
    return v if 0 <= v <= 3 else None


def lag1_pairs(
    turns: Sequence[Turn], depths: Sequence[Depth], a: str, b: str
) -> list[tuple[int, int]]:
    """(depth of an ``a`` turn, depth of the ``b`` turn immediately following it)."""
    pairs: list[tuple[int, int]] = []
    for i in range(len(turns) - 1):
        if turns[i].speaker_id == a and turns[i + 1].speaker_id == b:
            pairs.append((depths[i], depths[i + 1]))
    return pairs


def reciprocity_from_depths(
    turns: Sequence[Turn], depths: Sequence[Depth], a: str, b: str, min_pairs: int = 4
) -> float:
    """Lag-1 Pearson r of a's depth vs b's reply depth; 0.0 with < ``min_pairs`` or no variance."""
    pairs = lag1_pairs(turns, depths, a, b)
    if len(pairs) < min_pairs:
        return 0.0
    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=float)
    if x.std() == 0.0 or y.std() == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def mean_depth(turns: Sequence[Turn], depths: Sequence[Depth], speaker: str) -> float:
    own = [d for t, d in zip(turns, depths, strict=True) if t.speaker_id == speaker]
    return sum(own) / len(own) if own else 0.0


def asymmetry_from_depths(turns: Sequence[Turn], depths: Sequence[Depth], a: str, b: str) -> float:
    return abs(mean_depth(turns, depths, a) - mean_depth(turns, depths, b))


async def disclosure_reciprocity(
    transcript: Transcript, scorer: DisclosureScorer, a: str | None = None, b: str | None = None
) -> float:
    """Directed a -> b (defaults to participant order)."""
    a = a or transcript.participants[0]
    b = b or transcript.participants[1]
    turns = transcript.spoken()
    return reciprocity_from_depths(turns, await scorer.score(transcript), a, b)


async def disclosure_asymmetry(transcript: Transcript, scorer: DisclosureScorer) -> float:
    """|mean depth a - mean depth b|; symmetric."""
    a, b = transcript.participants[0], transcript.participants[1]
    turns = transcript.spoken()
    return asymmetry_from_depths(turns, await scorer.score(transcript), a, b)


__all__ = [
    "AFFECT",
    "FIRST_PERSON",
    "INTENSIFIERS",
    "Depth",
    "DisclosureScorer",
    "HeuristicDisclosureScorer",
    "LLMDisclosureScorer",
    "asymmetry_from_depths",
    "disclosure_asymmetry",
    "disclosure_reciprocity",
    "lag1_pairs",
    "mean_depth",
    "reciprocity_from_depths",
]
