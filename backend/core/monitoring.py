"""In-memory request metrics for the /metrics endpoint (course: monitoring.py)."""


class MetricsCollector:
    def __init__(self):
        self._latencies: list[float] = []
        self._requests = 0
        self._errors = 0
        self._cache_hits = 0
        self._input_tokens = 0
        self._output_tokens = 0

    def record_request(self, latency_ms: float, input_tokens: int, output_tokens: int,
                       error: bool = False, cached: bool = False) -> None:
        self._requests += 1
        self._latencies.append(latency_ms)
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        if error:
            self._errors += 1
        if cached:
            self._cache_hits += 1

    def _percentile(self, pct: float) -> float:
        if not self._latencies:
            return 0.0
        ordered = sorted(self._latencies)
        idx = min(len(ordered) - 1, int(pct / 100 * len(ordered)))
        return ordered[idx]

    def get_summary(self) -> dict:
        n = self._requests
        avg = sum(self._latencies) / n if n else 0.0
        error_rate = self._errors / n if n else 0.0
        cache_rate = self._cache_hits / n if n else 0.0
        return {
            "total_requests": n,
            "total_errors": self._errors,
            "error_rate": f"{error_rate:.1%}",
            "avg_latency_ms": round(avg, 2),
            "p99_latency_ms": round(self._percentile(99), 2),
            "cache_hit_rate": f"{cache_rate:.1%}",
            "total_input_tokens": self._input_tokens,
            "total_output_tokens": self._output_tokens,
        }
