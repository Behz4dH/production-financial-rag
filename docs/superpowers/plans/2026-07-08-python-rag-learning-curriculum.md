# Python/RAG Learning Curriculum — Session Plan

> **Note on execution model:** this plan is a teaching curriculum, not a software build — there is no code being shipped by a subagent. Execution means: the user and Claude work through modules together, live, in conversation. `subagent-driven-development`/`executing-plans` do not apply; this doc exists so sessions can pick up where they left off, and so each module has a concrete, checkable exercise instead of an open-ended "read the code" pass.

**Goal:** Take the learner from "comfortable with Python basics" to confidently understanding this repo's full architecture and the transferable Python/software-engineering patterns behind it, through 7 modules of lesson → exercise → checkpoint.

**Architecture:** Modules run in build order (mirrors `docs/superpowers/plans/plan1..plan6`). Each module is scoped to fit one session (Modules 3 and 4 may span two). Exercises are done live in this repo, on a scratch branch, never on `main`/the active feature branch's shipped code.

**Tech Stack:** Python 3.13, `uv`, `pydantic`/`pydantic-settings`, FastAPI, LangChain/LangGraph, ChromaDB, pytest.

## Global Constraints

- Exercises must not permanently modify the working repo's shipped behavior — use a scratch branch (`learning/moduleN`) or a throwaway file under `backend/tests/learning/`, and discard or leave unmerged when done.
- Every checkpoint must be gated by either (a) a runnable check (test passes, `make eval`/`make test` stays green) or (b) the learner correctly explaining the concept in their own words back to Claude — no module is "done" on vibes.
- Reference source of truth for architecture rationale: `docs/superpowers/specs/2026-07-02-financial-report-rag-design.md`. Reference source of truth for curriculum scope: `docs/superpowers/specs/2026-07-08-python-rag-learning-curriculum-design.md`.

---

## Session 0: Orientation

**Files to open together:** `backend/pyproject.toml`, `backend/Makefile`, `backend/.env.example`, top-level `app/`, `core/`, `rag/`, `eval/` directory listing.

**Lesson points:**
- Why `uv`/`pyproject.toml` over `requirements.txt` (lockfile, reproducibility).
- Walk the Makefile targets: `ingest`, `run`, `ui`, `test`, `eval` — what each invokes.
- `.env.example` → `.env`: why secrets/config never get committed, `groq_api_key` requirement.
- One-sentence tour of `app/` (web), `core/` (cross-cutting), `rag/` (ML core, framework-free), `eval/` (measurement) — the four-way split and why `rag/` is kept dependency-light.

**Exercise:**
1. Run `cd backend && uv sync`.
2. Copy `.env.example` to `.env`, get a Groq API key if not already present, fill it in.
3. Run `make ingest` — observe the summary output (files/chunks/dims).
4. Run `make run` in one terminal, then `curl -s -X POST localhost:8000/chat -H "Content-Type: application/json" -d '{"message": "What was Apple'"'"'s revenue in fiscal 2023?"}'` (or via `/docs` Swagger UI) in another.
5. Without reading `app/main.py` line-by-line yet, guess-and-verify: which file handles the route, which file builds the answer, which file loads settings? Write down guesses first, then check.

**Checkpoint (verbal):** Learner states, unprompted, which top-level package (`app`/`core`/`rag`/`eval`) they'd touch to: (a) add a new config value, (b) change how chunks are embedded, (c) add a new API field, (d) add a new benchmark question. All four must be correct.

---

## Session 1: Foundation & Config (mirrors Plan 1)

**Files:** `backend/core/config.py`, `backend/tests/` (whichever test file covers config, found via `grep -rl Settings backend/tests`).

**Lesson points:**
- `Settings(BaseSettings)`: one class, `.env`-loaded, typed fields — read every field group (LLM/embeddings, vector store, retrieval, security/reliability/limits, observability, app) and say in one sentence what each group governs.
- `@lru_cache` on `get_settings()`: memoized singleton — first call constructs, every later call returns the same object.
- `SettingsConfigDict(env_file=".env", extra="ignore")`: why `extra="ignore"` (tolerate unrelated env vars) vs `extra="forbid"`.
- `is_production` computed property pattern (derived, not stored, state).
- Rule: "nothing outside this module reads env vars" — a codebase-wide invariant, not just a config detail.

**Python/SE skills:** `pydantic-settings`, memoization via `functools.lru_cache`, computed properties, `pytest` fixtures for isolating env state.

**Exercise:**
1. On a scratch branch (`git checkout -b learning/module1`), add a new setting `max_answer_chars: int = 2000` to `Settings`.
2. Write a test in `backend/tests/` (new file `test_config_learning.py` is fine) asserting: (a) the default value is `2000`, (b) setting the env var `MAX_ANSWER_CHARS=500` and constructing a fresh `Settings()` picks it up. Use `monkeypatch.setenv` to isolate env changes.
3. Run `cd backend && uv run pytest tests/test_config_learning.py -v` — confirm it passes.
4. Discard the branch after the checkpoint (`git checkout <original-branch> && git branch -D learning/module1`) unless the learner wants to keep it.

**Checkpoint:** Learner explains, without looking at the code: why would two separate `Settings()` instances (instead of one shared singleton) be a real bug here, not just a style nitpick? (Expected answer shape: config could disagree mid-request, e.g. one part of the code sees old `top_k` and another sees an override — inconsistent behavior, hard-to-reproduce bugs.)

---

## Session 2: Providers & Ingestion (mirrors Plan 2)

**Files:** `backend/rag/providers/base.py`, `backend/rag/providers/factory.py`, `backend/rag/ingestion/loaders.py`, `metadata.py`, `chunking.py`, `pipeline.py`.

**Lesson points:**
- `base.py`: `Protocol`-based `LLM`/`EmbeddingsModel` — structural typing, not inheritance. Any object with the right methods satisfies the protocol.
- `factory.py`: `get_llm`/`get_embeddings` build from `Settings` — provider swap is a config value (`llm_provider`, `embedding_provider`), never a code change. `CacheBackedEmbeddings.from_bytes_store(...)`: wraps a base embedder so re-ingesting unchanged chunks skips recomputation — a decorator-like wrapping pattern, not inheritance.
- `loaders.py`: dispatch-by-extension, PyMuPDF table extraction vs plain text, one `Document` per page.
- `metadata.py`: LLM-extracted company/fiscal-year cached to `doc_metadata.json` — the single source of truth for entity attributes, resolved at **query time**, not baked into chunks.
- `chunking.py`: chunk metadata is deliberately `{source, page, chunk_id}` **only** — re-emphasize why (correcting a fiscal year later never requires re-embedding).
- `pipeline.py`: `ingest()` is idempotent — safe to re-run without duplicating data.

**Python/SE skills:** `typing.Protocol` vs `abc.ABC`, factory-function pattern, decorator/wrapper composition (`CacheBackedEmbeddings`), lazy loading/generators, idempotency as a design property you can test for.

**Exercise:**
1. On `learning/module2`, read `rag/providers/base.py` and write a second `Protocol`, `Reranker`, with one method `rerank(query: str, docs: list, top_n: int) -> list` — just the protocol, no implementation.
2. Write a test that a fake class implementing `rerank(...)` satisfies `isinstance(fake, Reranker)` when `Reranker` is `@runtime_checkable`. This demonstrates structural typing concretely.
3. Separately: call `ingest()` twice in a row on the same tiny sample dir (or read `pipeline.py` closely enough to explain, without running it, why chunk count doesn't double) — confirm/verify the idempotency claim by reading the guard condition in the code.
4. Discard branch after.

**Checkpoint:** Learner explains why `doc_metadata.json` is separate from chunk metadata, using the concrete scenario: "a 10-K's fiscal year was misdetected as 2022 instead of 2023 — walk me through fixing it under the current design vs. if fiscal year were stored per-chunk."

---

## Session 3a: Retrieval (mirrors Plan 3, part 1)

**Files:** `backend/rag/retrieval/store.py`, `entity_resolver.py`, `hybrid.py`, `reranker.py`, `rag/query.py`.

**Lesson points:**
- `store.py`: `VectorStore` interface, `similarity_search_with_score` (why exposing scores matters — refusal depends on it), metadata filtering by `source`.
- `entity_resolver.py`: resolves asked company+fiscal-year → matching `source` file(s) via `doc_metadata.json` (with alias matching) — **this is the first refusal gate**: no match ⇒ `N/A` before any retrieval happens.
- `hybrid.py`: `EnsembleRetriever` combining `BM25Retriever` (keyword) + vector (semantic), weighted — targets "embedding mismatch" (exact company names, financial jargon that embeddings blur).
- `reranker.py`: cross-encoder reorders `top_k` → `top_n` — targets retrieval noise, a second-pass precision filter.
- `rag/query.py`: façade — `answer(query, mode, deps)` where `mode ∈ {basic, hybrid, agentic}` is the one switch the API/eval use.

**Python/SE skills:** vector similarity search fundamentals, weighted ensemble/voting pattern, façade pattern (one simple interface hiding three different strategies), metadata-filtered querying.

**Exercise:**
1. On `learning/module3a`, write a small script (or test) that calls `rag/query.py`'s `answer()` (or the retriever directly, using existing `build_deps`) for the same question three times, once per `mode`, printing which chunks were retrieved for each. Compare `basic` vs `hybrid` retrieval order on a question you know the answer's page for.
2. Change `bm25_weight`/`vector_weight` in a scratch `Settings` override and re-run — observe how the ranked order shifts.
3. Deliberately bypass `entity_resolver` (call the retriever directly with a company that isn't in the corpus, skipping the resolver check) and observe: does the LLM now attempt an answer for a document that was never verified to match? This demonstrates why the gate exists — walk through what happens without it.

**Checkpoint:** Learner explains, in their own words, why `similarity_search_with_score` exposing a raw score is a prerequisite for the refusal logic — what would refusal have to fall back on without scores?

---

## Session 3b: Generation + Agentic loop (mirrors Plan 3, part 2)

**Files:** `backend/rag/agentic.py`, `backend/rag/generation/schema.py`, `generator.py`.

**Lesson points:**
- `agentic.py`: LangGraph `StateGraph`: `resolve-entity → retrieve → grade → route` — grade node is the **second refusal gate** (weak retrieval score ⇒ rewrite/retry up to `agentic_max_retries`, or refuse).
- `schema.py`/`generator.py`: `with_structured_output(RAGAnswer)` — `RAGAnswer{answer, citations, refused, computed_value, confidence}` as a typed contract the LLM must fill, so the refusal flag/citations can never get lost in free-text parsing the way regex-scraping an LLM's prose would risk.
- Prompt injection defense: user text passed as a bound variable/message, never `.format()`-ed into the template string — structural isolation, not just "we told the LLM to ignore instructions."

**Python/SE skills:** state machines for control flow (`LangGraph`), Pydantic models as an LLM I/O contract, why string-formatting user input into a template is a structural injection risk regardless of prompt wording.

**Exercise:**
1. On `learning/module3b`, read `agentic.py`'s grade node and write down, in your own pseudocode, the full decision tree (all branches: strong retrieval → generate; weak retrieval + retries left → rewrite/retry; weak retrieval + retries exhausted → refuse; no entity match → refuse immediately).
2. Find the exact line in `generator.py`/`prompt.py` where user input is bound as a variable rather than interpolated into a string, and rewrite it (in a scratch copy, not committed) the *unsafe* way (`f"...{user_input}..."` injected into the template) to see concretely what class of attack that would reopen.
3. Revert the unsafe scratch change; confirm you understand why it's wrong without needing to run an actual injection.

**Checkpoint:** Learner draws (verbally or on paper) the full agentic state graph from memory, correctly placing both refusal gates.

---

## Session 4a: API layer (mirrors Plan 4, part 1)

**Files:** `backend/app/main.py`, `app/models.py`, `app/rate_limit.py`, `app/mapping.py`.

**Lesson points:**
- App factory pattern (`create_app()`) + `lifespan`: heavy state (settings, providers, vector store) loaded once at startup, not per-request.
- Dependency injection via `app.state` (no `Depends()` boilerplate here — state lives on the app object, read per-request).
- Request flow line-by-line in the `/chat` handler: rate-limit → sanitize → cache lookup → retrieve/generate (with retry+fallback) → output validate/mask → cache set → metrics record → structured log.
- `_scoped_deps`: `dataclasses.replace`/`model_copy` to override `top_k`/`top_n` **per request** without mutating the shared `deps.settings` — critical under concurrency (shared object, many simultaneous requests).
- Cache key includes `mode:top_k:top_n:message` — why (an overridden request must not collide with a differently-configured cached answer).

**Python/SE skills:** FastAPI `lifespan`/app factory, `dataclasses.replace` and Pydantic `model_copy` for controlled immutability, reasoning about shared mutable state under concurrency, decorators (`@limiter.limit`).

**Exercise:**
1. On `learning/module4a`, add a new optional field to `ChatRequest` (e.g. `explain: bool = False`) and thread it through: `models.py` → the `/chat` handler → `mapping.py` (even if it does nothing yet but gets accepted and echoed back in a new `ChatResponse` field).
2. Write a test (`httpx`/FastAPI `TestClient`, following the existing test patterns in `backend/tests`) asserting two concurrent-style requests with different `top_k` values don't affect each other's results — i.e. exercise `_scoped_deps` directly or via two sequential calls checking `deps.settings.top_k` on the shared `app.state.query_deps` is unchanged after each.
3. Run `uv run pytest` for the touched test file, confirm green.

**Checkpoint:** Learner explains, with a concrete scenario, why mutating `deps.settings` in place (instead of `replace`) would leak one request's override into a concurrent request.

---

## Session 4b: Production concerns (mirrors Plan 4, part 2)

**Files:** `backend/core/cache.py`, `core/security.py`, `core/reliability.py`, `core/monitoring.py`, `core/logging.py`, `core/token_budget.py`.

**Lesson points:**
- `cache.py`: TTL response cache — hit/miss stats, key normalization.
- `security.py`: `SecurePipeline` = `InputSanitizer` (regex injection screen + structural isolation) → `PIIDetector` (mask before the LLM sees it) → optional `SecurityGuard` (LLM-as-guard, config-gated, `with_structured_output` so it can't fail open) → `OutputValidator` (re-check before returning). Defense-in-depth: each layer independently useful, composed in sequence.
- `reliability.py`: `with_retry` (exponential backoff + jitter), `CircuitBreaker` (closed/open/half-open state machine), `FallbackChain` (primary→fallback model escalation) — three distinct resilience patterns, each solving a different failure mode (transient blip vs. sustained outage vs. model-specific failure).
- `monitoring.py`/`logging.py`: `MetricsCollector` (requests/latency p99/tokens/cache hit-rate) + `JSONFormatter` structured logs — why JSON-per-line logging matters for log aggregation tooling.
- `token_budget.py`: real token counting via `tiktoken`, per-request cap enforced on the **assembled generation prompt**, not the raw user message (which is separately length-capped for input hygiene) — two different limits, two different purposes.

**Python/SE skills:** composing small single-purpose classes into a pipeline (defense-in-depth), state machines for resilience (circuit breaker), exponential backoff + jitter as a concrete algorithm (not just a buzzword), structured logging conventions, the difference between validating input length vs. budgeting assembled-prompt tokens.

**Exercise:**
1. On `learning/module4b`, write a test that mocks a provider call to fail twice then succeed, and asserts `with_retry` retries exactly twice before succeeding (inspect `reliability.py` for the exact retry signature/params first).
2. Write a second test that forces enough consecutive failures to trip the `CircuitBreaker` into the open state, then asserts a further call fails fast (no actual retry attempted) while open.
3. Read `token_budget.py` and explain in a comment (in your scratch test file, not committed to the real module) why the 2000-character `ChatRequest.message` cap and the 8000-token `max_context_tokens` cap are solving different problems.

**Checkpoint:** Learner explains all three resilience patterns (retry+jitter, circuit breaker, fallback chain) and gives a distinct real-world failure scenario each one is meant to handle.

---

## Session 5: Evaluation Harness (mirrors Plan 5)

**Files:** `backend/eval/golden.py`, `matching.py`, `metrics.py`, `runner.py`, `evaluate.py`, `backend/data/benchmark/questions.json`/`answers.json`.

**Lesson points:**
- `golden.py`: typed records for `question, schema, answers[], category, comment` — why typed records over raw dicts (IDE/type-checker help, self-documenting fields).
- `matching.py`: verdict logic — numeric tolerance (relative, not absolute), `N/A` matching, multi-valid-answer lists.
- `metrics.py`/`runner.py`/`evaluate.py`: per-category × per-mode aggregation; **why correct-refusal rate on the `hallucination` category is the headline metric**, not overall accuracy — the benchmark is ~60% `N/A` by design, so raw accuracy would reward a system that just says `N/A` to everything.

**Python/SE skills:** typed data records for test fixtures, designing small CLIs/reporting tools, choosing a headline metric deliberately (accuracy can lie when a dataset is imbalanced).

**Exercise:**
1. On `learning/module5`, add one new question+answer pair to the benchmark data (a simple retrieval-category question you can verify by hand from the source PDFs).
2. Run `make eval`, confirm the new item is scored and appears correctly in the per-category table.
3. Add a deliberately tricky case: a numeric golden answer that needs tolerance matching (e.g. `"$45.2 billion"` vs a model answer of `"$45,200 million"`) and verify `matching.py` handles it — if it doesn't, that's a real finding, not a scripted success; report what you find rather than forcing a pass.

**Checkpoint:** Learner explains why "just answer N/A to everything" would score deceptively well on raw accuracy here, and what refusal-correctness-on-`hallucination` actually measures instead.

---

## Session 6: Observability & Dashboard (mirrors Plan 6)

**Files:** `backend/core/tracing.py`, and whatever the Plan 6 trace/dashboard artifact currently is — check `docs/superpowers/plans/2026-07-08-plan6-trace-and-dashboard.md` at teaching time, since it may have evolved since this curriculum was written.

**Lesson points:**
- `@traceable` with name/tags/metadata; per-`/chat`-request trace including mode, model, cached flag, latency, token counts.
- No-op-when-no-API-key pattern: how an optional integration degrades gracefully via a single config check at the boundary, instead of scattering `if enabled` checks through the codebase.
- How a captured trace becomes a demo dashboard (whatever Plan 6 built).

**Python/SE skills:** decorators for cross-cutting concerns, designing a feature-detected optional integration.

**Exercise:**
1. On `learning/module6`, run one request with `LANGSMITH_API_KEY` unset and confirm (by reading `tracing.py`, and/or logs) that tracing is a true no-op, not a silent failure.
2. Add one new field to what gets traced (e.g. the resolved `sources` from the entity resolver) and confirm it shows up in the trace/dashboard output.

**Checkpoint:** Learner explains the no-op design and contrasts it with the alternative of `if settings.langsmith_tracing: ...` scattered at every call site — why is the chosen approach more maintainable?

---

## Self-review notes

- **Spec coverage:** all 7 modules from the design spec are present with lesson/skills/exercise/checkpoint. Module 3 and 4 split into 3a/3b and 4a/4b per the design's note that they may need two sessions.
- **No placeholders:** every exercise has concrete file paths and concrete actions; checkpoints are concrete questions/scenarios, not "TBD."
- **Consistency:** file paths cross-checked against actual repo listing as of 2026-07-08 (`rag/agentic.py` is top-level under `rag/`, not `rag/retrieval/agentic.py` — corrected from an earlier draft assumption).
