"""Hook factories that record metrics for bot tool calls and LLM calls.

Neither ``regulars.bot.tools`` nor ``regulars.llm.factory`` imports this package (subpackages
don't import each other's internals; see ``CLAUDE.md``). Instead each exposes a purely additive,
optional ``on_call`` hook — a plain ``Callable``, no dependency on ``regulars.observability``
types. Callers that want metrics (``regulars.wiring.build_system``) build the hook here and pass
it in.
"""

from __future__ import annotations

from collections.abc import Callable

from regulars.observability.metrics import BOT_TOOL_CALLS, BOT_TOOL_LATENCY, LLM_CALLS, LLM_LATENCY


def bot_tool_hook() -> Callable[[str, float, bool], None]:
    """Hook for ``BotTools(..., on_call=...)``: records a call count and latency per tool name."""

    def _on_call(tool: str, duration_seconds: float, success: bool) -> None:
        BOT_TOOL_CALLS.inc(tool=tool, status="ok" if success else "error")
        BOT_TOOL_LATENCY.observe(duration_seconds, tool=tool)

    return _on_call


def llm_call_hook(role: str) -> Callable[[float, bool], None]:
    """Hook for ``llm_from_env(role, on_call=...)``: records a call count and latency for `role`."""

    def _on_call(duration_seconds: float, success: bool) -> None:
        LLM_CALLS.inc(role=role, status="ok" if success else "error")
        LLM_LATENCY.observe(duration_seconds, role=role)

    return _on_call
