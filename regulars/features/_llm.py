"""One-call JSON batching shared by the LLM-backed classifiers (private to regulars.features).

Each LLM classifier numbers every spoken turn, asks for one JSON array of labels in ONE
``LLMClient.complete`` call, and falls back per turn to the heuristic label when the model's
answer is unusable. No classifier ever makes a call per turn.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from typing import Any

from regulars.core.llm import LLMClient, LLMMessage
from regulars.core.transcript import Turn

log = logging.getLogger(__name__)


def number_turns(turns: Sequence[Turn]) -> str:
    return "\n".join(f"{i + 1}. [{t.speaker_id}] {t.text.strip()}" for i, t in enumerate(turns))


def extract_json(raw: str) -> Any:
    """Parse the first JSON value in ``raw``; tolerates code fences and surrounding prose."""
    s = raw.strip()
    starts = [i for i in (s.find("["), s.find("{")) if i >= 0]
    if not starts:
        raise ValueError("no JSON value in LLM response")
    start = min(starts)
    closer = "]" if s[start] == "[" else "}"
    end = s.rfind(closer)
    if end < start:
        raise ValueError("unterminated JSON value in LLM response")
    return json.loads(s[start : end + 1])


def labels_from_payload(payload: Any, n: int) -> list[Any]:
    """Accept ``[...]`` or ``{"labels": [...]}``; require exactly ``n`` entries."""
    items = payload.get("labels") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("LLM response is not a list of labels")
    if len(items) != n:
        raise ValueError(f"LLM returned {len(items)} labels for {n} turns")
    return items


async def batch_labels[T](
    client: LLMClient,
    *,
    system: str,
    turns: Sequence[Turn],
    fallback: Sequence[T],
    normalize: Callable[[Any], T | None],
    per_turn_tokens: int = 8,
) -> list[T]:
    """Label every turn in one call. ``normalize(item)`` maps a raw entry to a label or None.

    Any entry ``normalize`` rejects takes the heuristic ``fallback`` label; an unparsable or
    mis-sized response falls back wholesale (logged, never raised — a bad model answer must not
    take feature extraction down).
    """
    if not turns:
        return []
    user = (
        "Turns:\n"
        + number_turns(turns)
        + f"\n\nReturn a JSON array with exactly {len(turns)} entries, one per turn, in order."
    )
    raw = await client.complete(
        system,
        [LLMMessage(role="user", content=user)],
        temperature=0.0,
        max_tokens=max(256, per_turn_tokens * len(turns) + 64),
    )
    try:
        items = labels_from_payload(extract_json(raw), len(turns))
    except (ValueError, json.JSONDecodeError) as exc:
        log.warning("LLM label batch unusable (%s); using heuristic labels", exc)
        return list(fallback)
    out: list[T] = []
    for item, fb in zip(items, fallback, strict=True):
        label = normalize(item)
        out.append(fb if label is None else label)
    return out
