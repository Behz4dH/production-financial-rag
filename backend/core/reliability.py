"""Retry with exponential backoff, retryability classification, model fallback."""

import random
import time

_RETRYABLE_4XX = {408, 425, 429}


def is_retryable(exc: Exception) -> bool:
    """False for deterministic client errors (4xx except timeout/rate-limit):
    retrying those only adds latency and cost. Exceptions without a
    status_code are assumed transient."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500:
        return status in _RETRYABLE_4XX
    return True


def with_retry(fn, *, max_retries: int, base_delay: float = 1.0,
               max_delay: float = 30.0, exceptions: tuple = (Exception,),
               retry_if=None):
    """Call fn() up to max_retries times with exponential backoff and jitter.
    ``retry_if(exc) -> bool`` short-circuits non-retryable exceptions."""
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
                time.sleep(delay * (0.5 + random.random()))
    raise last


def call_with_fallback(primary, fallback):
    """Run primary(); on any exception run fallback(). Raise if both fail."""
    try:
        return primary()
    except Exception:
        return fallback()
