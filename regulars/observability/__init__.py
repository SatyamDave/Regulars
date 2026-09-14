"""Structured logging and in-process metrics for the running system.

This is operational observability (logs and counters an operator watches), distinct from
``regulars.db.repo.write_audit``'s ``audit_log`` table, which is a decision record ("why this
group, why this removal") kept for reproducibility and user-facing legibility, not for
debugging latency or throughput.

Public surface:

- :func:`regulars.observability.logging_config.configure_logging` and
  :func:`regulars.observability.logging_config.bind_context` / ``get_context`` — JSON logging
  with request-scoped context (``request_id``, ``user_id``) via ``contextvars``.
- :class:`regulars.observability.middleware.RequestLoggingMiddleware` — one structured log line
  per HTTP request, plus an ``X-Request-Id`` response header.
- :mod:`regulars.observability.metrics` — ``Counter``, ``Histogram``, the global ``REGISTRY``,
  and ``render_prometheus_text`` for a ``GET /metrics`` route.
- :mod:`regulars.observability.instrumented` — hooks that record bot-tool-call and LLM-call
  metrics, built to be passed into ``regulars.bot.tools.BotTools``'s optional ``on_call`` and
  ``regulars.llm.factory.llm_from_env``'s optional ``on_call`` without either module importing
  this package.
"""

from __future__ import annotations
