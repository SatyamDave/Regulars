"""Test-only helpers: a FakeLLM responder that answers each batched classifier prompt."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from regulars.core.llm import LLMMessage

_TURN_LINE = re.compile(r"^\d+\. \[", re.M)


def turn_count(messages: Sequence[LLMMessage]) -> int:
    return len(_TURN_LINE.findall(messages[0].content))


def scripted_responder(system: str, messages: Sequence[LLMMessage]) -> str:
    """Route on the system prompt: questions -> 'none', topics -> one topic each, depth -> 1."""
    n = turn_count(messages)
    if "question type" in system:
        return json.dumps(["none"] * n)
    if "segment" in system:
        return json.dumps(list(range(n)))
    if "self-disclosure" in system:
        return json.dumps([1] * n)
    raise AssertionError(f"unexpected system prompt: {system[:60]}")
