"""MetricsCollector aggregates request stats for /metrics."""

from core.monitoring import MetricsCollector


def test_summary_with_no_requests_is_zeroed():
    s = MetricsCollector().get_summary()
    assert s["total_requests"] == 0
    assert s["error_rate"] == "0.0%"
    assert s["avg_latency_ms"] == 0.0


def test_records_latency_tokens_errors_and_cache():
    m = MetricsCollector()
    m.record_request(100.0, 10, 20, error=False, cached=False)
    m.record_request(300.0, 5, 15, error=True, cached=True)
    s = m.get_summary()
    assert s["total_requests"] == 2
    assert s["total_errors"] == 1
    assert s["error_rate"] == "50.0%"
    assert s["avg_latency_ms"] == 200.0
    assert s["cache_hit_rate"] == "50.0%"
    assert s["total_input_tokens"] == 15
    assert s["total_output_tokens"] == 35


def test_p99_latency_is_high_percentile():
    m = MetricsCollector()
    for i in range(100):
        m.record_request(float(i), 0, 0)
    # p99 of 0..99 is ~98-99
    assert m.get_summary()["p99_latency_ms"] >= 98.0
