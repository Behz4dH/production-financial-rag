"""Retry-with-backoff and model fallback (no sleeps in tests: base_delay=0)."""

import pytest

from core.reliability import call_with_fallback, is_retryable, with_retry


class _APIError(Exception):
    """Shaped like an SDK API error (e.g. groq.APIStatusError)."""

    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_with_retry_succeeds_after_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    assert with_retry(flaky, max_retries=3, base_delay=0.0) == "ok"
    assert calls["n"] == 3


def test_with_retry_reraises_after_exhaustion():
    def always_fail():
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        with_retry(always_fail, max_retries=2, base_delay=0.0)


def test_fallback_used_when_primary_fails():
    def primary():
        raise RuntimeError("primary down")

    assert call_with_fallback(primary, lambda: "from-fallback") == "from-fallback"


def test_fallback_not_used_when_primary_ok():
    assert call_with_fallback(lambda: "primary-ok", lambda: "fallback") == "primary-ok"


def test_is_retryable_classifies_by_status_code():
    # Deterministic client errors: retrying can never succeed.
    assert is_retryable(_APIError(400)) is False
    assert is_retryable(_APIError(404)) is False
    assert is_retryable(_APIError(413)) is False
    assert is_retryable(_APIError(422)) is False
    # Transient: timeouts, rate limits, server errors.
    assert is_retryable(_APIError(408)) is True
    assert is_retryable(_APIError(429)) is True
    assert is_retryable(_APIError(500)) is True
    assert is_retryable(_APIError(503)) is True
    # No status code (network errors, bugs): assume transient.
    assert is_retryable(ConnectionError("reset")) is True


def test_with_retry_gives_up_immediately_on_non_retryable():
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise _APIError(400)

    with pytest.raises(_APIError):
        with_retry(bad_request, max_retries=3, base_delay=0.0, retry_if=is_retryable)
    assert calls["n"] == 1  # no second attempt against a deterministic failure


def test_with_retry_still_retries_transient_errors_with_retry_if():
    calls = {"n": 0}

    def rate_limited_then_ok():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _APIError(429)
        return "ok"

    assert with_retry(rate_limited_then_ok, max_retries=3, base_delay=0.0,
                      retry_if=is_retryable) == "ok"
    assert calls["n"] == 3


def test_with_retry_raises_on_invalid_max_retries():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "ok"

    with pytest.raises(ValueError, match="max_retries must be >= 1"):
        with_retry(fn, max_retries=0, base_delay=0.0)

    assert calls["n"] == 0  # fn was never called
