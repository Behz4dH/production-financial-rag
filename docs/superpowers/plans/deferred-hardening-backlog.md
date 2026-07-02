# Deferred Hardening Backlog

Non-blocking items surfaced during reviews, to be addressed in the plan noted.
These were intentionally out of scope for the foundation slice.

## From Plan 1 final review (cache & settings)
- **Cache eviction / unbounded growth** — `core/cache.py` only evicts an entry when
  its own key is read; a set-once/never-read key is never reclaimed. Add a max-entry
  bound or periodic sweep **before the cache backs the `/chat` endpoint (Plan 4)**.
- **Monotonic clock** — `core/cache.py` uses `time.time()` for TTL; switch to
  `time.monotonic()` (immune to wall-clock adjustments) when touching the cache in Plan 4.
- **Settings cache isolation in tests** — as more modules use `get_settings()`, add a
  fixture calling `get_settings.cache_clear()` between tests to avoid stale-settings
  pollution (Plan 2+ test infra).
- **Minor (optional):** cache key does not collapse internal whitespace (affects hit
  rate only); `JSONFormatter` stamps format-time rather than `record.created`.
