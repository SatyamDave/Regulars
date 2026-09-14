"""Question typing and the follow-up rate (docs/technical-architecture.md §4).

Huang, Yeomans, Brooks, Minson & Gino (JPSP 2017): daters who asked more follow-up questions
elicited more second dates. Each spoken turn gets one ``QuestionType``; the directed feature
``follow_up_rate`` is follow-ups as a share of the asker's turns, and ``question_gini`` is the
imbalance in who asks (heavy imbalance = interview, not conversation).

Two ``QuestionClassifier`` implementations share one contract: the regex/lexicon heuristic is
the cheap first pass and the test path; ``LLMQuestionClassifier`` labels all turns in ONE call.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, get_args

from regulars.core.llm import LLMClient
from regulars.core.transcript import Transcript, Turn
from regulars.features._llm import batch_labels
from regulars.features._text import content_lemmas, jaccard

QuestionType = Literal["none", "new_topic", "follow_up", "mirror", "rhetorical"]
QUESTION_TYPES: tuple[QuestionType, ...] = get_args(QuestionType)

# Labels that count as actually asking something (rhetorical questions do not seek an answer).
ASKING: frozenset[str] = frozenset({"new_topic", "follow_up", "mirror"})


class QuestionClassifier(Protocol):
    """One label per spoken turn (``transcript.spoken()`` order)."""

    async def classify(self, transcript: Transcript) -> list[QuestionType]: ...


_WH_START = re.compile(
    r"^(?:who|whom|whose|what|which|when|where|why|how)\b|"
    r"^(?:do|does|did|are|is|was|were|have|has|had|can|could|would|will|should|shall|may|might)"
    r"\s+(?:you|we|they|i|he|she|it|there|that|this|anyone|anybody)\b"
)
_FOLLOW_UP_START = re.compile(r"^(?:so|and|wait|how come|why|but)\b")
_MIRROR_PATTERNS = re.compile(
    r"\b(?:what|how|and) about you\b|\byou\?$|\byourself\?$|\bsame (?:question|for you)\b|"
    r"\bback at you\b|\bhow about yourself\b"
)
_RHETORICAL = re.compile(
    r"\bwho knows\b|\bwho cares\b|\bwho knew\b|\bwhat can you do\b|\bwhy not\b|\bhow about that\b|"
    r"\bguess what\b|\bwhat are the odds\b|\bcan you believe\b|\bam i right\b|\bwhat a\b|"
    r"\bwhy do i even\b|\bwho would(?:'ve| have)? thought\b|\bwho am i (?:to|kidding)\b"
)
_TAG_QUESTION = re.compile(
    r",\s*(?:right|no|yeah|eh|huh|isn't it|aren't they|don't you|doesn't it|wouldn't you|"
    r"you know|ok|okay)\?\s*$"
)


def is_question(text: str) -> bool:
    """Surface test: a question mark, or a wh-/aux-inverted opening."""
    s = text.strip().lower()
    if not s:
        return False
    if "?" in s:
        return True
    return bool(_WH_START.match(s))


def _last_other_turn(turns: Sequence[Turn], i: int) -> Turn | None:
    me = turns[i].speaker_id
    for j in range(i - 1, -1, -1):
        if turns[j].speaker_id != me:
            return turns[j]
    return None


@dataclass(frozen=True)
class HeuristicQuestionClassifier:
    """Regex + lexicon typing.

    Order of tests for a question turn: rhetorical lexicon / tag question -> mirror (echoes the
    question the other speaker just asked, or "what about you?") -> follow-up (shares >= 1
    content lemma with the previous other-speaker turn, or opens with so/and/wait/how come/why)
    -> new_topic.
    """

    mirror_overlap: float = 0.5

    def classify_turns(self, turns: Sequence[Turn]) -> list[QuestionType]:
        return [self._label(turns, i) for i in range(len(turns))]

    async def classify(self, transcript: Transcript) -> list[QuestionType]:
        return self.classify_turns(transcript.spoken())

    def _label(self, turns: Sequence[Turn], i: int) -> QuestionType:
        text = turns[i].text.strip().lower()
        if not is_question(text):
            return "none"
        if _RHETORICAL.search(text) or _TAG_QUESTION.search(text):
            return "rhetorical"
        prev = _last_other_turn(turns, i)
        if prev is None:
            return "new_topic"
        prev_text = prev.text.strip().lower()
        if _MIRROR_PATTERNS.search(text) and is_question(prev_text):
            return "mirror"
        mine, theirs = content_lemmas(text), content_lemmas(prev_text)
        if is_question(prev_text) and mine and jaccard(mine, theirs) >= self.mirror_overlap:
            return "mirror"
        if mine & theirs or _FOLLOW_UP_START.match(text):
            return "follow_up"
        return "new_topic"


_SYSTEM = """You label each turn of a two-person conversation with exactly one question type:
- "none": the turn asks nothing.
- "new_topic": a question that opens a subject not raised in the other person's previous turn.
- "follow_up": a question that asks more about what the other person just said.
- "mirror": the speaker bounces the question they were just asked back ("what about you?").
- "rhetorical": a question that does not seek an answer.
Answer with a JSON array of strings only."""


@dataclass(frozen=True)
class LLMQuestionClassifier:
    """Labels every turn in one ``LLMClient`` call; invalid entries fall back to the heuristic."""

    client: LLMClient
    fallback: HeuristicQuestionClassifier = HeuristicQuestionClassifier()

    async def classify(self, transcript: Transcript) -> list[QuestionType]:
        turns = transcript.spoken()
        return await batch_labels(
            self.client,
            system=_SYSTEM,
            turns=turns,
            fallback=self.fallback.classify_turns(turns),
            normalize=_normalize,
        )


def _normalize(item: Any) -> QuestionType | None:
    if isinstance(item, str):
        s = item.strip().lower().replace("-", "_").replace(" ", "_")
        if s in QUESTION_TYPES:
            return s
    return None


def follow_up_rate_from_labels(
    turns: Sequence[Turn], labels: Sequence[QuestionType], asker: str
) -> float:
    """Follow-up questions by ``asker`` / turns by ``asker``; 0.0 when the asker never spoke."""
    own = [lab for t, lab in zip(turns, labels, strict=True) if t.speaker_id == asker]
    if not own:
        return 0.0
    return sum(lab == "follow_up" for lab in own) / len(own)


async def follow_up_rate(
    transcript: Transcript, asker: str, target: str, clf: QuestionClassifier
) -> float:
    """Directed: share of ``asker``'s turns to ``target`` that are follow-up questions."""
    dyad = (
        transcript
        if set(transcript.participants) == {asker, target}
        else transcript.restrict(asker, target)
    )
    labels = await clf.classify(dyad)
    return follow_up_rate_from_labels(dyad.spoken(), labels, asker)


def gini(values: Sequence[float]) -> float:
    """Gini coefficient with the small-sample n/(n-1) correction (all-in-one-hand -> 1.0)."""
    n = len(values)
    total = float(sum(values))
    if n < 2 or total <= 0:
        return 0.0
    abs_diffs = sum(abs(x - y) for x in values for y in values)
    g = abs_diffs / (2 * n * total)
    return min(1.0, g * n / (n - 1))


def question_counts(
    transcript: Transcript, labels: Sequence[QuestionType] | None = None
) -> dict[str, int]:
    """Asking questions per participant (rhetorical excluded)."""
    turns = transcript.spoken()
    if labels is None:
        labels = HeuristicQuestionClassifier().classify_turns(turns)
    counts = dict.fromkeys(transcript.participants, 0)
    for t, lab in zip(turns, labels, strict=True):
        if lab in ASKING:
            counts[t.speaker_id] += 1
    return counts


def question_gini(transcript: Transcript, labels: Sequence[QuestionType] | None = None) -> float:
    """Imbalance in who asks: 0 = balanced, 1 = one person asks everything."""
    return gini(list(question_counts(transcript, labels).values()))


__all__ = [
    "ASKING",
    "QUESTION_TYPES",
    "HeuristicQuestionClassifier",
    "LLMQuestionClassifier",
    "QuestionClassifier",
    "QuestionType",
    "follow_up_rate",
    "follow_up_rate_from_labels",
    "gini",
    "is_question",
    "question_counts",
    "question_gini",
]
