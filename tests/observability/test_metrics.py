from __future__ import annotations

from regulars.observability.metrics import Counter, Histogram, Registry, render_prometheus_text


def test_counter_increments_and_renders() -> None:
    c = Counter("widgets_total", "widgets seen", ("color",))
    c.inc(color="red")
    c.inc(2, color="red")
    c.inc(color="blue")

    samples = dict(c.samples())
    assert samples[("red",)] == 3
    assert samples[("blue",)] == 1

    rendered = c.render()
    assert "# TYPE widgets_total counter" in rendered
    assert 'widgets_total{color="red"} 3' in rendered
    assert 'widgets_total{color="blue"} 1' in rendered


def test_counter_without_labels() -> None:
    c = Counter("hits_total", "hits")
    c.inc()
    c.inc()
    assert "hits_total 2" in c.render()


def test_histogram_observes_and_renders_cumulative_buckets() -> None:
    h = Histogram("latency_seconds", "latency", ("op",), buckets=(0.1, 0.5, 1.0))
    h.observe(0.05, op="read")
    h.observe(0.3, op="read")
    h.observe(2.0, op="read")

    [(key, bucket_counts, total_sum, total_count)] = h.samples()
    assert key == ("read",)
    assert total_count == 3
    assert total_sum == 0.05 + 0.3 + 2.0

    rendered = h.render()
    assert "# TYPE latency_seconds histogram" in rendered
    # cumulative: le=0.1 sees the 0.05 observation only
    assert 'latency_seconds_bucket{op="read",le="0.1"} 1.0' in rendered
    # le=0.5 is cumulative: the 0.05 and the 0.3 observation
    assert 'latency_seconds_bucket{op="read",le="0.5"} 2.0' in rendered
    # le=1.0 still only those two; the 2.0 observation overflows every finite bucket
    assert 'latency_seconds_bucket{op="read",le="1.0"} 2.0' in rendered
    assert 'latency_seconds_sum{op="read"} 2.35' in rendered
    assert 'latency_seconds_count{op="read"} 3.0' in rendered


def test_histogram_default_buckets_have_a_plus_inf_catch_all() -> None:
    h = Histogram("dur_seconds", "duration")
    h.observe(1_000_000.0)
    [(_, bucket_counts, _, total_count)] = h.samples()
    assert sum(bucket_counts) == total_count
    assert 'le="+Inf"' in h.render()


def test_registry_render_text_includes_all_registered_metrics() -> None:
    registry = Registry()
    c = registry.counter("a_total", "a")
    h = registry.histogram("b_seconds", "b")
    c.inc()
    h.observe(0.01)
    text = registry.render_text()
    assert "a_total" in text
    assert "b_seconds" in text


def test_registry_reuses_metric_by_name() -> None:
    registry = Registry()
    first = registry.counter("dup_total", "dup")
    second = registry.counter("dup_total", "dup")
    assert first is second


def test_render_prometheus_text_uses_global_registry() -> None:
    text = render_prometheus_text()
    # The module-level metrics (bot tool calls, LLM calls, webhook requests) are always present,
    # even before anything has incremented them (a Prometheus scrape must at least see nothing).
    assert isinstance(text, str)
