"""Tests for the in-memory ResponseCache."""

import time

from core.cache import ResponseCache


def test_miss_returns_none():
    assert ResponseCache().get("anything") is None


def test_set_then_get_hit():
    cache = ResponseCache()
    cache.set("hello", "world")
    assert cache.get("hello") == "world"


def test_key_normalization_is_case_and_whitespace_insensitive():
    cache = ResponseCache()
    cache.set("Hello World", "answer")
    assert cache.get("  hello world  ") == "answer"


def test_entry_expires_after_ttl():
    cache = ResponseCache(ttl_seconds=0)
    cache.set("q", "a")
    time.sleep(0.01)
    assert cache.get("q") is None


def test_expired_entry_is_evicted():
    cache = ResponseCache(ttl_seconds=0)
    cache.set("q", "a")
    time.sleep(0.01)
    cache.get("q")  # triggers eviction
    assert cache.stats["cached_entries"] == 0


def test_stats_track_hits_and_misses():
    cache = ResponseCache()
    cache.get("missing")  # miss
    cache.set("q", "a")
    cache.get("q")  # hit
    cache.get("q")  # hit
    stats = cache.stats
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == "66.7%"
    assert stats["cached_entries"] == 1


def test_stats_hit_rate_with_no_requests():
    stats = ResponseCache().stats
    assert stats["hit_rate"] == "0.0%"
    assert stats["cached_entries"] == 0
