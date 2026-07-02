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

## From Plan 2 final review (store & ingestion) — address in Plan 3
- **`store.count()` private API** — `ChromaStore.count()` reaches `self._store._collection.count()`;
  a `langchain-chroma` bump could break it. Fine on the pin; revisit if upgrading.
- **Score type coercion** — `similarity_search_with_score` returns Chroma's raw distance,
  which may be `numpy.float32`. Wrap in `float(score)` in Plan 3 before comparing against
  `refusal_score_threshold`, so the refusal gate is portable.
- **Stale-vector drift on corpus change** — the pipeline upserts by `chunk_id` and rewrites
  the docstore fresh, so it is idempotent for a *fixed* corpus (proven by test). But if a
  re-ingest ever yields fewer chunks for a page, or a source is removed, orphaned vectors
  linger in Chroma and drift from the docstore. Add a `reset()` / delete-by-source hook to
  `VectorStore` in Plan 3 (call before re-ingest, or clear-then-add).
- **Cosmetic:** `load_text` uses `errors="replace"` (kept — robustness for real filings);
  empty-text+table page leaves a leading blank line in the chunk.
