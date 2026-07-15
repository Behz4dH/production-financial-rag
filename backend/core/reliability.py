"""Reliability helpers: retry with exponential backoff + model fallback
(course: error_handling.py). Kept simple — three small functions."""

import random
import time

# 4xx errors that are transient despite being client errors.
_RETRYABLE_4XX = {408, 425, 429}


def is_retryable(exc: Exception) -> bool:
    """Whether retrying can plausibly succeed.

    SDK API errors (groq/httpx style) carry a ``status_code``: deterministic
    client errors (400 bad request, 413 too large, 422 …) will fail identically
    every attempt, so retrying them only adds latency and cost. Timeouts, rate
    limits, and server errors are transient. Exceptions without a status code
    (network resets, timeouts surfacing as OSError) are assumed transient.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500:
        return status in _RETRYABLE_4XX
    return True


def with_retry(fn, *, max_retries: int, base_delay: float = 1.0,
               max_delay: float = 30.0, exceptions: tuple = (Exception,),
               retry_if=None):
    """Call fn() up to max_retries times with exponential backoff + jitter.

    ``retry_if(exc) -> bool`` short-circuits: a non-retryable exception is
    re-raised immediately instead of burning the remaining attempts.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except exceptions as exc:
            if retry_if is not None and not retry_if(exc):
                raise
            last = exc
            if attempt < max_retries - 1 and base_delay > 0:
                delay = min(base_delay * (2 ** attempt), max_delay)
                time.sleep(delay * (0.5 + random.random()))  # jitter
    raise last


def call_with_fallback(primary, fallback):
    """Run primary(); on any exception run fallback(). Raise if both fail."""
    try:
        return primary()
    except Exception:
        return fallback()
