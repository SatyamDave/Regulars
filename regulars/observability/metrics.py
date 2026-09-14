"""Minimal in-process metrics registry: ``Counter``, ``Histogram``, a global ``REGISTRY``, and
``render_prometheus_text`` for a ``GET /metrics`` route.

No dependency on ``prometheus_client``: this is a small subset (counters, histograms, text
exposition format) sufficient for scraping, implemented over stdlib only.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Mapping, Sequence

DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    float("inf"),
)


def _label_key(labelnames: Sequence[str], labels: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(str(labels.get(name, "")) for name in labelnames)


def _format_labels(labelnames: Sequence[str], key: Sequence[str]) -> str:
    if not labelnames:
        return ""
    pairs = ",".join(f'{name}="{value}"' for name, value in zip(labelnames, key, strict=True))
    return "{" + pairs + "}"


class Counter:
    """A monotonically increasing value, optionally partitioned by label values."""

    def __init__(self, name: str, documentation: str, labelnames: Sequence[str] = ()) -> None:
        self.name = name
        self.documentation = documentation
        self.labelnames = tuple(labelnames)
        self._values: dict[tuple[str, ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = _label_key(self.labelnames, labels)
        with self._lock:
            self._values[key] += amount

    def samples(self) -> list[tuple[tuple[str, ...], float]]:
        with self._lock:
            return list(self._values.items())

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.documentation}", f"# TYPE {self.name} counter"]
        for key, value in self.samples():
            lines.append(f"{self.name}{_format_labels(self.labelnames, key)} {value}")
        return "\n".join(lines)


class Histogram:
    """Observation counts bucketed by ``buckets`` (upper bounds, inclusive), plus sum and count."""

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> None:
        self.name = name
        self.documentation = documentation
        self.labelnames = tuple(labelnames)
        self.buckets = tuple(buckets)
        self._bucket_counts: dict[tuple[str, ...], list[float]] = defaultdict(
            lambda: [0.0] * len(self.buckets)
        )
        self._sums: dict[tuple[str, ...], float] = defaultdict(float)
        self._counts: dict[tuple[str, ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def observe(self, value: float, **labels: str) -> None:
        key = _label_key(self.labelnames, labels)
        with self._lock:
            self._sums[key] += value
            self._counts[key] += 1
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    self._bucket_counts[key][i] += 1
                    break

    def samples(self) -> list[tuple[tuple[str, ...], list[float], float, float]]:
        """One ``(label_key, per_bucket_counts, sum, count)`` row per label combination seen."""
        with self._lock:
            keys = set(self._counts.keys())
            return [
                (key, list(self._bucket_counts[key]), self._sums[key], self._counts[key])
                for key in keys
            ]

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.documentation}", f"# TYPE {self.name} histogram"]
        for key, bucket_counts, total_sum, total_count in self.samples():
            cumulative = 0.0
            for bound, count in zip(self.buckets, bucket_counts, strict=True):
                cumulative += count
                le = "+Inf" if bound == float("inf") else str(bound)
                labels = _format_labels((*self.labelnames, "le"), (*key, le))
                lines.append(f"{self.name}_bucket{labels} {cumulative}")
            base_labels = _format_labels(self.labelnames, key)
            lines.append(f"{self.name}_sum{base_labels} {total_sum}")
            lines.append(f"{self.name}_count{base_labels} {total_count}")
        return "\n".join(lines)


class Registry:
    """Owns named counters and histograms and renders all of them as Prometheus text."""

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, documentation: str, labelnames: Sequence[str] = ()) -> Counter:
        with self._lock:
            existing = self._counters.get(name)
            if existing is None:
                existing = Counter(name, documentation, labelnames)
                self._counters[name] = existing
            return existing

    def histogram(
        self,
        name: str,
        documentation: str,
        labelnames: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> Histogram:
        with self._lock:
            existing = self._histograms.get(name)
            if existing is None:
                existing = Histogram(name, documentation, labelnames, buckets)
                self._histograms[name] = existing
            return existing

    def render_text(self) -> str:
        with self._lock:
            counters = list(self._counters.values())
            histograms = list(self._histograms.values())
        parts = [c.render() for c in counters] + [h.render() for h in histograms]
        return "\n".join(parts) + "\n" if parts else ""


REGISTRY = Registry()

# Bot tool calls (regulars.bot.tools.BotTools, via the optional ``on_call`` hook).
BOT_TOOL_CALLS = REGISTRY.counter(
    "regulars_bot_tool_calls_total", "Bot tool calls by tool name and outcome", ("tool", "status")
)
BOT_TOOL_LATENCY = REGISTRY.histogram(
    "regulars_bot_tool_call_duration_seconds", "Bot tool call latency", ("tool",)
)

# LLM calls (regulars.llm.factory.llm_from_env, via the optional ``on_call`` hook).
LLM_CALLS = REGISTRY.counter(
    "regulars_llm_calls_total", "LLM calls by role and outcome", ("role", "status")
)
LLM_LATENCY = REGISTRY.histogram(
    "regulars_llm_call_duration_seconds", "LLM call latency by role", ("role",)
)

# Webhook / HTTP requests (regulars.observability.middleware.RequestLoggingMiddleware).
WEBHOOK_REQUESTS = REGISTRY.counter(
    "regulars_webhook_requests_total", "HTTP requests by route and status", ("route", "status")
)


def render_prometheus_text() -> str:
    """Render every registered counter and histogram in Prometheus text exposition format."""
    return REGISTRY.render_text()
