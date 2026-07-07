"""Reliability helpers: retry with exponential backoff + model fallback
(course: error_handling.py). Kept simple — two small functions."""

import random
import time


def with_retry(fn, *, max_retries: int, base_delay: float = 1.0,
               max_delay: float = 30.0, exceptions: tuple = (Exception,)):
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except exceptions as exc:
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
