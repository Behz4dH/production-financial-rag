"""In-memory response cache with per-entry TTL expiration."""

import hashlib
import time
from typing import Optional


class ResponseCache:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self._cache: dict[str, dict] = {}
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str) -> str:
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        key = self._make_key(query)
        entry = self._cache.get(key)
        if entry is not None:
            if (time.time() - entry["timestamp"]) < self.ttl:
                self._hits += 1
                return entry["response"]
            del self._cache[key]
        self._misses += 1
        return None

    def set(self, query: str, response: str) -> None:
        self._cache[self._make_key(query)] = {
            "response": response,
            "query": query,
            "timestamp": time.time(),
        }

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1%}",
            "cached_entries": len(self._cache),
        }
