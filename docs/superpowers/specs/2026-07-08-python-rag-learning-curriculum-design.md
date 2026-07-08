# Python/RAG Learning Curriculum — Design Spec

**Date:** 2026-07-08
**Author:** Behzad (with Claude)
**Status:** Approved design → ready for session-plan breakdown
**Purpose:** Teach the user (comfortable with Python basics, new to production RAG/LLM-app architecture) to deeply understand this repository's code and concepts, well enough to design and build a similar system independently — and to come out of it a stronger general Python developer, not just someone who has read this codebase.

---

## 1. Context & goals

This repo (`production-rag`) was built as 6 sequential plans (`docs/superpowers/plans/plan1..plan6`), each adding a layer: config → providers/ingestion → retrieval/generation → API → eval → observability. That build order is itself a good teaching order — it's how a real engineer would reason about the system, dependency by dependency.

**Learner profile:** comfortable with Python syntax/functions/classes; has not done much with async, Protocols/structural typing, pydantic, FastAPI internals, or LLM-application patterns (RAG, agentic loops, structured output, prompt injection defense).

**Learning mode:** guided walkthrough + rebuild exercises — for each module, explain the concept and real code, then the learner writes or modifies a small piece of code themselves before moving on. Not a "read-only" tour, and not a from-scratch parallel project (too slow, forfeits reusing this project's own reasoning/tradeoffs as teaching material).

**Success criteria:**
1. For every module, the learner can answer *what does this piece do, how do you use it, what does it depend on* (this repo's own design bar) without re-reading the source.
2. The learner has personally written and run code exercising every general Python concept in §3, not just read about it.
3. By the end, the learner could sketch the architecture of a similar RAG system from memory and explain every production concern (caching, rate limiting, retries, tracing, eval) and why it's there.
4. Each module ends with a checkpoint (a question set or a small runnable test) the learner must pass before advancing.

---

## 2. Non-goals

- Not teaching general Python fundamentals (syntax, control flow, basic OOP) — learner already has this.
- Not building a second parallel codebase from scratch (Approach C, rejected — see conversation).
- Not covering the frontend (React/Vite chat UI) — backend/Python only, per user's stated goal ("better python developer").
- Not exhaustively covering every file — some small/mechanical files (e.g. `app/mapping.py`, `rag/retrieval/docstore.py`) are mentioned in passing within their module rather than getting dedicated treatment.

---

## 3. Curriculum structure

7 modules, in build order. Each module has four parts, always in this order:

1. **Lesson** — walk the real code in this repo (file + line references), explain the concept and *why it's shaped this way* (tradeoffs the design doc already made explicit).
2. **Python/SE skills** — the transferable concepts this module teaches, independent of RAG.
3. **Rebuild exercise** — the learner writes or modifies a small, runnable piece of code (a stripped-down reimplementation, an extension, or a targeted bug/change) — never just reading.
4. **Checkpoint** — a short question set or a runnable check (e.g. "make this test pass," "explain why X would break if Y changed") gating advancement to the next module.

### Module 0 — Orientation
- **Lesson:** repo layout (`app/`, `core/`, `rag/`, `eval/`), `pyproject.toml`, `uv`, `Makefile` targets (`ingest`, `run`, `eval`, `test`), `.env.example`.
- **Skills:** Python packaging basics (`uv`/`pyproject.toml` vs `requirements.txt`), why env-based config matters, Makefile-as-task-runner convention.
- **Exercise:** run `make ingest && make run && make eval` end-to-end; trace one `/chat` request through the Makefile/README to identify which module handles it.
- **Checkpoint:** learner can point to the file responsible for each of: settings, LLM calls, vector store, API routes, eval scoring.

### Module 1 — Foundation & Config (mirrors Plan 1)
- **Files:** `core/config.py`, relevant tests.
- **Lesson:** single `Settings(BaseSettings)` object, `.env` loading, `@lru_cache` singleton (`get_settings()`), fail-fast validation, "nothing outside this module reads env vars" rule.
- **Skills:** `pydantic-settings`/`BaseSettings`, `@lru_cache` as a memoized-singleton pattern, `SettingsConfigDict(extra="ignore")`, why centralizing config prevents drift, `pytest` fixture basics.
- **Exercise:** add a new setting (e.g. a toggle), wire it through `Settings`, write a test asserting the default and an env-override both work.
- **Checkpoint:** explain why `get_settings()` uses `@lru_cache` instead of a module-level global, and what would break if two different `Settings()` instances existed at once.

### Module 2 — Providers & Ingestion (mirrors Plan 2)
- **Files:** `rag/providers/base.py`, `rag/providers/factory.py`, `rag/ingestion/loaders.py`, `metadata.py`, `chunking.py`, `pipeline.py`.
- **Lesson:** `Protocol`-based `LLMProvider`/`EmbeddingsProvider` (swap providers via config, not code), `CacheBackedEmbeddings` (never recompute an unchanged chunk's vector), PDF loading (PyMuPDF table extraction vs plain text), chunk metadata deliberately minimal (`{source, page, chunk_id}` — entity fields live in `doc_metadata.json`, not denormalized onto chunks) and why that separation avoids re-embedding on metadata correction.
- **Skills:** `Protocol`/structural typing vs ABCs, factory-function pattern, decorator-style wrapping (`CacheBackedEmbeddings.from_bytes_store`), lazy iteration/generators (`DirectoryLoader`/`lazy_load`), idempotency as a design property.
- **Exercise:** write a `Protocol` for a new provider type (or add a second embedding provider branch to `_build_base_embeddings`) and a test that swaps providers via a fake `Settings`.
- **Checkpoint:** explain why entity metadata (company, fiscal year) is *not* stored on chunks, and what would happen if it were and you needed to correct a misdetected fiscal year.

### Module 3 — Retrieval & Generation (mirrors Plan 3) — meatiest module
- **Files:** `rag/retrieval/store.py`, `entity_resolver.py`, `hybrid.py`, `reranker.py`, `rag/agentic.py`, `rag/query.py`, `rag/generation/schema.py`, `generator.py`.
- **Lesson:** vector similarity search + scores (why exposing scores enables refusal), BM25 ⊕ vector weighted ensemble (the "embedding mismatch" failure mode for exact names/jargon), cross-encoder rerank, the entity-resolver-as-refusal-gate (refuse *before* retrieval if no doc matches company/year), the agentic LangGraph loop (`resolve → retrieve → grade → [rewrite/retry] → generate/refuse`), structured LLM output (`with_structured_output(RAGAnswer)`) so refusal/citations can't be lost in free-text parsing, prompt-injection defense (binding user text as a variable vs `.format()`-ing it into a template).
- **Skills:** vector/keyword search concepts, ensemble/weighted-voting pattern, state machines (`LangGraph StateGraph`), Pydantic models as an LLM output schema/contract, structural string-injection defense, façade pattern (`retriever.py`/`rag/query.py` exposing one `mode` switch over 3 strategies).
- **Exercise:** trace one query through `rag/query.py` by hand for each of the 3 modes (`basic`, `hybrid`, `agentic`); then modify `hybrid.py`'s weights and observe retrieval order change on a sample query; then deliberately break the refusal gate (bypass `entity_resolver`) and observe a hallucinated answer, to feel *why* the gate exists.
- **Checkpoint:** explain the refusal gate's two checkpoints (entity resolution before retrieval, grade node after retrieval) and why both are needed.

### Module 4 — API & Production concerns (mirrors Plan 4)
- **Files:** `app/main.py`, `app/models.py`, `app/rate_limit.py`, `core/cache.py`, `core/security.py`, `core/reliability.py`, `core/monitoring.py`, `core/logging.py`, `core/token_budget.py`.
- **Lesson:** FastAPI app factory + `lifespan` (load heavy state once, not per-request), dependency injection via `app.state`, request flow (rate-limit → sanitize → cache lookup → retrieve/generate → validate output → cache set → metrics), `dataclasses.replace` for per-request settings overrides *without* mutating shared state (and why that matters under concurrency), retry+jitter / circuit breaker / fallback chain (`core/reliability.py`), defense-in-depth security pipeline (`InputSanitizer` → `PIIDetector` → `[LLM guard]` → `OutputValidator`), structured JSON logging, token budgeting with `tiktoken`.
- **Skills:** FastAPI internals (`lifespan`, middleware, `Depends`-free DI via `app.state`), decorators (`@limiter.limit`, `@asynccontextmanager`), immutability patterns (`dataclasses.replace`, `model_copy`), resilience patterns (retry/backoff/jitter, circuit breaker, fallback chain), structured logging as a production concern, defense-in-depth composition.
- **Exercise:** add a new field to `ChatRequest`/`ChatResponse` and wire it through `main.py`/`mapping.py`; then write a test that two concurrent requests with different `top_k` overrides don't leak into each other (exercising `_scoped_deps`); then trip the circuit breaker in a test by mocking repeated provider failures.
- **Checkpoint:** explain why `_scoped_deps` uses `replace`/`model_copy` instead of mutating `deps.settings` directly, with a concrete concurrency scenario where mutation would leak.

### Module 5 — Evaluation Harness (mirrors Plan 5)
- **Files:** `eval/golden.py`, `matching.py`, `metrics.py`, `runner.py`, `evaluate.py`.
- **Lesson:** golden-dataset loading into typed records, verdict/matching logic (numeric tolerance, `N/A` matching, multi-valid answers), why **correct-refusal rate on the `hallucination` category** is the headline metric (not overall accuracy), per-category × per-mode reporting.
- **Skills:** typed records (dataclasses/NamedTuples) for structured test data, designing a small CLI/reporting tool, aggregation logic, why a domain's "success metric" should be chosen deliberately (headline metric vs supporting metrics).
- **Exercise:** add a new golden question + answer, run `make eval`, and confirm it's scored correctly; then intentionally add a question with a tricky matching edge case (e.g. a numeric answer needing tolerance) and verify `matching.py` handles it, adjusting if not.
- **Checkpoint:** explain why refusal-correctness on `hallucination` is the headline metric instead of raw accuracy, given the benchmark's ~60% N/A composition.

### Module 6 — Observability & Dashboard (mirrors Plan 6)
- **Files:** `core/tracing.py`, whatever pipeline-trace/dashboard artifact Plan 6 produced (check `docs/superpowers/plans/2026-07-08-plan6-trace-and-dashboard.md` at teaching time for current state).
- **Lesson:** `@traceable` tracing with metadata (mode, model, tokens, latency), why tracing is no-op without an API key (graceful degradation of an optional integration), how a trace turns into a demo dashboard.
- **Skills:** decorators for cross-cutting concerns (tracing), designing optional/pluggable integrations (feature-detected via config, not hard dependency).
- **Exercise:** run one traced request and one untraced request; inspect what data a trace captures; extend the trace metadata with one new field.
- **Checkpoint:** explain how the no-op path is implemented and why that pattern (vs. `if enabled: ...` scattered everywhere) is preferable.

---

## 4. Cross-cutting Python skill arc

Skills threaded across multiple modules, so the learner sees them reinforced, not just once:

- **Typing:** type hints throughout → `Protocol` (M2) → Pydantic models as contracts (M3) → dataclasses + `replace`/`model_copy` (M4).
- **Config/state management:** `@lru_cache` singleton (M1) → `app.state` DI (M4).
- **Caching/memoization:** `CacheBackedEmbeddings` (M2) → response `TTL` cache (M4).
- **Resilience:** idempotent pipelines (M2) → retry/circuit-breaker/fallback (M4).
- **Testing:** pytest basics (M1) → mocked-provider unit tests (M2-M4) → eval-as-testing (M5).
- **Composability/façades:** provider factory (M2) → retrieval mode façade (M3) → app factory (M4).

## 5. Format & pacing

- Multi-session curriculum; each module is roughly one session (Module 3 and 4 may need two given their weight).
- Progress tracked via `TaskCreate`/checkpoints inside sessions; the written plan (next artifact, via `writing-plans`) breaks each module into concrete session-level tasks with exercise instructions and checkpoint questions spelled out.
- The learner's rebuild exercises happen **in this repo** (small branches/scratch files), not a separate project — reusing real code and its existing tests as ground truth.

## 6. Testing/verification strategy for the curriculum itself

Each module's checkpoint must be objectively gate-able: either a question the learner must answer correctly in their own words, or a runnable exercise (new test passes, `make eval`/`make test` still green after the change). No module is marked complete on "sounds right" — either a test passes or the explanation is verified against the actual code/design doc reasoning.

---

## 7. Open items for the implementation plan

- Exact per-module exercise scaffolding (where rebuild exercises live — e.g. `backend/tests/learning/` scratch area vs throwaway branches) — to be decided in the session plan.
- Whether checkpoints are asked verbally in-session or written down anywhere — default: verbal/in-conversation, no artifact needed.
