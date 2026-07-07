"""Retry-with-backoff and model fallback (no sleeps in tests: base_delay=0)."""

import pytest

from core.reliability import call_with_fallback, with_retry


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


def test_with_retry_raises_on_invalid_max_retries():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "ok"

    with pytest.raises(ValueError, match="max_retries must be >= 1"):
        with_retry(fn, max_retries=0, base_delay=0.0)

    assert calls["n"] == 0  # fn was never called
