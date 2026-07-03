# Financial-Report RAG — Design Spec

**Date:** 2026-07-02
**Author:** Behzad (with Claude)
**Status:** Approved design → ready for implementation planning
**Purpose:** A production-grade, hallucination-resistant Retrieval-Augmented Generation system over SEC 10-K / annual-report PDFs, built as an AI/ML engineering interview portfolio piece. It must clone-and-run quickly, demonstrate clean separation of concerns, and — above all — show measured retrieval and answer quality on an adversarial benchmark.

---

## 1. Context & Goals

### 1.1 What we are building
A docs/knowledge Q&A RAG whose corpus is **20 real SEC 10-K / annual-report PDFs** (the "ERC" set) accompanied by a **categorized golden benchmark** (`questions.json` / `answers.json`, 40 Q/A tagged `hallucination`, `tricky`, `compare`, `retrieval`).

The benchmark is **deliberately adversarial**:
- ~60% of correct answers are `"N/A"` — the company is not in the corpus, or the report is for a different fiscal year than the question asks. These test **refusal / hallucination-resistance**, not retrieval.
- `tricky`: compute a financial ratio (Net Profit Margin, ROA, Quick Ratio, Operating Margin, EPS, Free Cash Flow = Operating Cash Flow − Capital Expenditures) from underlying line items → numerical reasoning.
- `compare`: "which company had higher X" across multiple documents, sometimes in different currencies (JPY vs CHF vs USD) → multi-document retrieval + reasoning.
- `retrieval`: answer on specific pages, occasionally with multiple valid values.

### 1.2 Primary engineered goal
**Hallucination-resistant, citation-grounded answering.** The headline metric is the **correct-refusal rate** on the `hallucination` category: the system must answer `"N/A"` when the corpus does not support an answer (wrong company or wrong fiscal year), rather than fabricate.

### 1.3 Scope decisions (locked)
- **Reasoning:** grounding + refusal are first-class; ratio/compare numerical reasoning is **best-effort** (attempted by the LLM from retrieved line items, reported per-category, not the pass/fail bar).
- **Corpus:** the ERC 10-Ks are **the project corpus** (copied into the repo `data/`), used for both the live demo and the eval.
- **Providers:** abstracted; default **Groq Llama** (LLM) + **local HuggingFace `sentence-transformers`** (embeddings). Swappable to OpenAI via config.
- **Vector store:** persistent **Chroma** behind a `VectorStore` interface; **pgvector** documented as a drop-in scale-up path (not implemented now).
- **Retrieval:** hybrid (BM25 + vector) → cross-encoder rerank → grounded generation, PLUS a selectable **agentic LangGraph loop** (grade → rewrite → retry → generate/refuse).
- **Interface:** FastAPI backend + minimal React (Vite) chat frontend.
- **Evaluation:** full harness — retrieval metrics + LLM-as-judge answer quality — reported per category and per retrieval mode.

### 1.4 Non-goals (YAGNI)
- Multi-tenant auth / user accounts.
- Runtime document upload (corpus is batch-ingested offline).
- Multimodal/vision retrieval (ColPali) — noted as a limitation, not built.
- Horizontal scaling infra (documented via the pgvector seam, not built).
- Fine-tuning any model.

### 1.5 Quality bar
Match or exceed the reference project `C:\Projects\production_rag_api`: `pydantic-settings` config with `@lru_cache`, `.env.example`, typed Pydantic request/response models, in-memory TTL response cache, `slowapi` rate limiting, LangSmith tracing (no-op without a key), pytest suite, clear module boundaries. The reference is an unfinished chat skeleton with **no RAG**; this project delivers the RAG it never built, with stronger separation of concerns.

---

## 2. Architecture (Approach A — layered RAG package + API + frontend)

```
production-rag/
├── backend/
│   ├── pyproject.toml            # uv-managed, Python 3.13
│   ├── .env.example
│   ├── Makefile                  # ingest / run / ui / test / eval
│   ├── app/                      # web layer (FastAPI)
│   │   ├── main.py               # app factory + lifespan (load store/providers once)
│   │   ├── routes.py             # /chat, /health, /metrics
│   │   ├── models.py             # Pydantic request/response/error models
│   │   ├── deps.py               # DI: settings, store, cache, retriever
│   │   └── rate_limit.py         # slowapi limiter
│   ├── core/                     # cross-cutting concerns
│   │   ├── config.py             # pydantic-settings Settings + get_settings()
│   │   ├── cache.py              # TTL response cache (reused from reference)
│   │   ├── security.py           # SecurePipeline: sanitize→PII→[guard]→validate
│   │   ├── monitoring.py         # JSON metrics: requests/latency(p99)/tokens/cache
│   │   ├── token_budget.py       # tiktoken token counting + per-request budget
│   │   ├── logging.py            # structured JSON logging setup
│   │   ├── reliability.py        # retry+jitter, circuit breaker, model fallback chain
│   │   └── tracing.py            # LangSmith: per-request tracing with metadata
│   ├── rag/                      # framework-free ML core (independently testable)
│   │   ├── providers/
│   │   │   ├── base.py           # LLMProvider, EmbeddingsProvider protocols
│   │   │   └── factory.py        # build providers from settings (Groq/HF default)
│   │   ├── ingestion/
│   │   │   ├── loaders.py        # pypdf text + pdfplumber tables, by extension
│   │   │   ├── metadata.py       # extract company name + fiscal year per doc
│   │   │   ├── chunking.py       # splitter; chunk metadata = {source, page, chunk_id} only
│   │   │   └── pipeline.py       # ingest(data_dir): load→chunk→embed→persist (idempotent)
│   │   ├── retrieval/
│   │   │   ├── store.py          # VectorStore interface + ChromaStore (filter by source)
│   │   │   ├── entity_resolver.py # company/year → source via doc_metadata.json (refusal lever)
│   │   │   ├── hybrid.py         # BM25 ⊕ vector ensemble (weighted)
│   │   │   ├── reranker.py       # cross-encoder rerank (top_k → top_n)
│   │   │   ├── agentic.py        # LangGraph: resolve→retrieve→grade→[rewrite]→generate/refuse
│   │   │   └── retriever.py      # façade; mode = basic | hybrid | agentic
│   │   └── generation/
│   │       ├── prompt.py         # refusal-first grounded prompt templates
│   │       └── generator.py      # format cited context → LLM → answer + citations
│   ├── eval/
│   │   ├── golden.py             # load ERC questions.json/answers.json
│   │   ├── matching.py           # numeric tolerance + N/A + multi-valid-answer matching
│   │   ├── evaluators.py         # {key,score} evaluators: correctness/refusal/faithfulness
│   │   ├── metrics.py            # hit-rate@k, MRR; LLM-judge faithfulness/groundedness
│   │   ├── evaluate.py           # local harness: run all modes → per-category table
│   │   └── langsmith_eval.py     # optional: versioned dataset + evaluate() experiments
│   ├── data/
│   │   ├── docs/                 # the 20 ERC 10-K PDFs (copied in)
│   │   ├── benchmark/            # questions.json, answers.json
│   │   └── chroma/               # persisted index (gitignored)
│   └── tests/                    # pytest (mocked providers; no API key needed)
├── frontend/                     # minimal Vite + React chat page
│   ├── package.json
│   └── src/ (ChatPage, MessageList, CitationChips, api client)
├── docs/
│   └── superpowers/specs/2026-07-02-financial-report-rag-design.md
├── README.md                     # 60-second quickstart, architecture diagram, eval table
└── .gitignore
```

**Design principle:** `rag/` is pure and framework-free — retrieval and generation are unit-testable without FastAPI. Each sub-package answers "what does it do / how do you use it / what does it depend on" in one breath. `app/` (web), `core/` (cross-cutting), and `eval/` (measurement) are separated from the ML core.

---

## 3. Components (responsibility · interface · dependencies)

### 3.1 `rag/providers`
- **base.py** — `LLMProvider` protocol (`complete(messages) -> str`, exposes `model_name`); `EmbeddingsProvider` protocol (`embed_documents`, `embed_query`).
- **factory.py** — `get_llm(settings)`, `get_embeddings(settings)`. Built on `init_chat_model(model, model_provider=..., max_retries=...)` (course `working_with_llms.py` pattern) so provider swap is a config value, not code. Default: Groq `llama-3.1-8b-instant` + local `sentence-transformers` `BAAI/bge-small-en-v1.5`. Adding OpenAI/Anthropic is a config change, not a core change.
- **embeddings** are wrapped in **`CacheBackedEmbeddings` (LocalFileStore)** (course `embeddings_deep.py`) so re-ingestion never recomputes an unchanged chunk's vector.
- Depends on: `langchain` (`init_chat_model`), `langchain-groq` / `langchain-huggingface`, `langchain-classic` (embedding cache); `core.config`.

### 3.2 `rag/ingestion`
- **loaders.py** — dispatch by extension: PDF → **PyMuPDF** (`find_tables()` renders genuine ≥2×2 tables as **Markdown**, and their regions are subtracted from the narrative to avoid duplication); `.md`/`.txt` → plain read. One Document **per page** (`{source, page}`); folder streamed lazily. PyMuPDF detects the borderless financial tables that line-based extractors miss (pdfplumber found 0; PyMuPDF finds dozens per filing, incl. currency in table headers).
- **metadata.py** — best-effort LLM extraction (`with_structured_output`) of **company name, aliases, fiscal year, period-end, reporting currency** per document, cached to **`doc_metadata.json` — the single source of truth for entity attributes** (NOT copied onto chunks). Resolved at query time; correcting an entry never requires re-embedding.
- **chunking.py** — `RecursiveCharacterTextSplitter` with explicit separators (tuned size/overlap for financial text). Chunk metadata carries **only `{source, page, chunk_id}`** — a back-reference to the source document, deliberately **not** the entity fields (those stay in `doc_metadata.json`; denormalizing them onto chunks would force a re-ingest on every metadata edit).
- **pipeline.py** — `ingest(data_dir, store)`: load → chunk → embed → persist to Chroma; also builds/persists the BM25 corpus. Idempotent (safe re-run). Prints a summary (files, chunks, dims).
- Runs offline via `python -m rag.ingestion` / `make ingest`.

### 3.3 `rag/retrieval`
- **store.py** — `VectorStore` interface (`add`, `similarity_search`, **`similarity_search_with_score`**, `count`); `ChromaStore` persistent impl. Exposes **scores** (distance→similarity) so a low top-score can trigger refusal, and **metadata filtering by `source`** so a query can be constrained to specific documents. Optional MMR for diversity. pgvector adapter documented as the swap point.
- **entity_resolver.py** — resolves the asked **company + fiscal year → matching `source` file(s)** by looking them up in `doc_metadata.json` (with alias matching). This is the refusal lever: if no source matches the asked company/year, the answer is `N/A` *before any retrieval*; if a source matches, retrieval is filtered to it by `source`. Entity attributes are read from the single source of truth at query time — never denormalized onto chunks.
- **hybrid.py** — weighted ensemble of BM25 (keyword) + vector (semantic) retrievers. Targets the "embedding mismatch" failure mode (financial jargon, exact company names).
- **reranker.py** — cross-encoder (`BAAI/bge-reranker-base`) reorders top_k candidates to top_n. Targets "retrieval noise."
- **agentic.py** — LangGraph `StateGraph`: `resolve-entity → retrieve → grade → route`. The **entity resolver + grade node form the refusal gate**: if `doc_metadata.json` has no source for the asked company/fiscal-year, refuse (`N/A`) up front; otherwise retrieve (filtered by `source`) and grade retrieval **score**. Routes to `rewrite→retry` (bounded by `max_retries`), `generate`, or `refuse`. Adapted from the course Part 6 agentic pattern + the `error_handling.py` conditional-retry graph.
- **retriever.py** — façade exposing `retrieve(query, mode)` where `mode ∈ {basic, hybrid, agentic}`, so the API and eval switch strategies uniformly.

### 3.4 `rag/generation`
- **prompt.py** — refusal-first grounded template: "Answer ONLY from the provided report excerpts. If the company or the requested fiscal year is not present, answer exactly `N/A`. Cite company · fiscal year · page for every figure." Separate template variant for ratio computation (best-effort). User text is passed as a **bound variable / message**, never `.format()`-ed into the template (structural injection defense).
- **generator.py** — formats retrieved chunks with source tags → calls LLM via **`with_structured_output(RAGAnswer)`** (course `output_parsers_final.py`) returning a typed `RAGAnswer{answer: str, citations: list[Citation], refused: bool, computed_value: float | None, confidence: str}`. Structured output means the refusal flag and citations can't be lost in free-text parsing.

### 3.5 `core`
- **config.py** — single `Settings(BaseSettings)`, `.env`-loaded, `@lru_cache` singleton, `is_production` property, `extra="ignore"`. Nothing outside this module reads env vars.
- **cache.py** — TTL response cache with hit/miss stats (reused from reference, key normalized by query). Semantic (embedding-similarity) caching documented as an enhancement (course `cost_optimization.py` `SemanticCache`), not built by default.
- **security.py** — a full **defense-in-depth `SecurePipeline`** (matches and extends the course `security_patterns.py`), wired through the `providers` abstraction (not hardcoded to OpenAI), composed of four units:
  1. `InputSanitizer` — prompt-injection regex screen + delimiter/brace neutralization, **plus structural isolation** (user text bound as a variable, never `.format()`-ed into the template).
  2. `PIIDetector` — detect + mask email/phone/SSN/credit-card/IP on the **input path before the text reaches the LLM**; mask uses a `{type: replacement}` map (no repetitive if/elif).
  3. `SecurityGuard` — optional LLM-as-guard using **`with_structured_output`** so parsing can't fail-open. **Config-gated (`enable_llm_guard`, default off)** to avoid an extra LLM call per request.
  4. `OutputValidator` — re-checks output for PII leakage + harmful patterns before returning.
  Every block/mask emits a structured security event + a `/metrics` counter. Flow: `sanitize → input-PII-mask → [LLM guard] → generate → output-validate`.
- **monitoring.py** — `JSONFormatter` + `MetricsCollector` (course `monitoring.py`): total requests, errors, error_rate, latency (avg **+ p99, exceeding the course's avg-only**), tokens in/out, cache hit-rate; `get_summary()` maps 1:1 to the reference `HealthResponse`/`MetricsResponse`. Exposed via `/metrics`.
- **token_budget.py** — `TokenBudget` (course `cost_optimization.py`): real token counting via **`tiktoken`**, per-request cap, and usage stats; rejects over-budget requests early.
- **logging.py** — **structured JSON logging** via `JSONFormatter` (one JSON object per line, for log aggregation) honoring `log_level`; includes request id, mode, latency, status per request.
- **reliability.py** — provider-call resilience, expanded to match `error_handling.py`: `with_retry` (**exponential backoff + jitter**), a **`CircuitBreaker`** (closed/open/half-open) around provider calls, and a **`FallbackChain`** (primary→fallback model escalation).
- **tracing.py** — LangSmith setup (`@traceable` with **name + tags + metadata**, course `langsmith_setup.py`); **every `/chat` request is traced with metadata** (thread_id, retrieval mode, model, cached flag, latency, token counts). No-op when `LANGSMITH_API_KEY` is absent.

### 3.6 `app`
- **main.py** — FastAPI app factory; lifespan loads settings, providers, and the persisted store once; wires CORS for the React dev origin.
- **routes.py** — `POST /chat`, `GET /health`, `GET /metrics`, Swagger at `/`.
- **models.py** — `ChatRequest {message, thread_id, mode?}`, `ChatResponse {response, citations[], model_used, mode, cached, processing_time_ms, timestamp}`, `HealthResponse`, `MetricsResponse`, `ErrorResponse`.
- **deps.py** — DI providers for settings/store/cache/retriever.
- **rate_limit.py** — slowapi limiter (`rate_limit` from config).

### 3.7 `eval`
- **golden.py** — parse `questions.json` + `answers.json` into typed records (`question, schema, answers[], category, comment`).
- **matching.py** — verdict logic: numeric tolerance (relative), `N/A` matching, and multiple-valid-answer support (golden answers are lists).
- **evaluators.py** — evaluators returning `{key, score}` (course `testing_patterns.py` interface): `correctness` (numeric/N-A match vs golden), `refusal_correctness` (did it correctly answer `N/A`?), and LLM-as-judge `faithfulness`/`groundedness`. Same functions feed both the local harness and LangSmith.
- **metrics.py** — retrieval: hit-rate@k, MRR (did a chunk from the right company/year get retrieved?). Answer: aggregates the evaluators above.
- **evaluate.py** — local harness: run the benchmark across `basic | hybrid | agentic`, print a **per-category × per-mode** table. Headline: correct-refusal rate on `hallucination`.
- **langsmith_eval.py** — optional, key-gated (course `testing_patterns.py`): push the golden set as a **versioned LangSmith dataset**, run `evaluate()` once **per retrieval mode** with `experiment_prefix` so the three modes are comparable side-by-side in the LangSmith dashboard (basic vs hybrid vs agentic). Elevates eval from a one-off script to versioned, comparable experiments.

---

## 4. Data flow

### 4.1 Ingestion (offline)
`data/docs/*.pdf` → loaders (text + tables) → metadata (company, fiscal year) → chunk (+metadata) → embed (local HF) → persist Chroma at `data/chroma/`; build BM25 corpus. Idempotent.

### 4.2 Query (online, per request)
```
POST /chat {message, thread_id, mode?}
 → rate-limit → sanitize input → cache lookup (hit ⇒ return)
 → retriever(mode):
     basic:   vector top_k
     hybrid:  BM25 ⊕ vector → rerank(top_k → top_n)
     agentic: LangGraph(retrieve → grade/refusal-gate → [rewrite→retry ≤N] → generate | refuse)
 → generation: refusal-first grounded prompt + cited context → LLM
 → output validation / PII mask → cache set → metrics update
 → 200 {response, citations[], model_used, mode, cached, processing_time_ms}
```
LangSmith tracing wraps retrieval + generation when `LANGSMITH_API_KEY` is present; otherwise a no-op.

---

## 5. Error handling & production concerns
- Global exception handler → structured `ErrorResponse` (never a raw 500).
- Provider timeout/failure → retry up to `max_retries` with **exponential backoff**, then primary→fallback model.
- Empty or low-relevance retrieval, or company/fiscal-year mismatch → explicit grounded **`N/A`** (not a hallucination).
- Rate-limit exceeded → HTTP 429.
- Input sanitization + **input-side PII masking (before the LLM)** + output re-check, on by default.
- Config validation fails fast at startup (missing required `groq_api_key` → clear error).

### 5.1 Production-Ready API checklist coverage
Every item below is a first-class deliverable of the API (Docker intentionally excluded per scope):

| Feature | What it does | Where |
| --- | --- | --- |
| LangSmith tracing | Every request traced with metadata (mode, model, tokens, latency) | `core/tracing.py`, wraps `/chat` |
| Input sanitization | Blocks prompt-injection attempts | `core/security.py` (`InputSanitizer`) |
| PII detection/masking | Redacts emails, SSNs, cards **before** the LLM | `core/security.py` (`PIIDetector`), input path |
| Error handling + retries | Exponential backoff + jitter, circuit breaker, model fallbacks | `core/reliability.py` |
| Response caching | In-memory cache for duplicate calls | `core/cache.py` |
| Rate limiting | Per-IP throttling via slowapi | `app/rate_limit.py` |
| Structured logging | JSON logs for production aggregation | `core/logging.py` |
| Metrics collection | Request count, latency, token usage | `core/monitoring.py`, `GET /metrics` |
| Health checks | `/health` endpoint | `app/routes.py` |
| Docker deployment | Optional (`Dockerfile` + compose), not required | repo root (optional) |

---

## 6. Testing strategy
pytest, with providers mocked so the suite runs **without any API key**:
- Unit: chunking (metadata preserved), metadata extraction, cache (reuse reference tests), hybrid fusion ordering, reranker ordering, agentic routing/refusal logic (mocked grader), config validation, eval matching (numeric tolerance, N/A, multi-valid), security pipeline (injection block + PII mask), reliability (retry backoff, circuit-breaker state transitions, fallback chain), token budget.
- Mocking follows the course `testing_patterns.py` pattern: `Mock()` LLM returning `AIMessage(content=...)`, `assert_called_once()`.
- API: `/chat`, `/health`, `/metrics` via `httpx` with mocked retriever/generator.
- The ML core (`rag/`) is tested hardest — it is the differentiator.

---

## 7. Runnability (interview-critical)
```
cd backend
uv sync
cp .env.example .env          # add GROQ_API_KEY
make ingest                   # build the index from data/docs
make run                      # FastAPI on :8000
make ui                       # React dev server (separate terminal)
make eval                     # run the benchmark → per-category table
make test                     # pytest
```
Sample corpus (ERC 10-Ks + benchmark) ships in `data/`, so it works out of the box. Optional `Dockerfile` + `docker-compose.yml`. README includes a 60-second quickstart, an architecture diagram, and the eval results table (the headline artifact).

---

## 8. Dependencies (backend)
`fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `python-dotenv`, `slowapi`, `langchain`, `langchain-core`, `langgraph`, `langchain-groq`, `langchain-huggingface`, `langchain-chroma`, `langchain-classic` (`CacheBackedEmbeddings`, BM25), `chromadb`, `sentence-transformers`, `rank-bm25`, `pypdf`, `pdfplumber`, `numpy`, `langsmith`, `tiktoken`; dev: `pytest`, `httpx`. Frontend: `vite`, `react`.

---

## 9. Risks & mitigations
- **Table extraction fidelity** — 10-K tables are hard; `pdfplumber` mitigates but won't be perfect. Mitigation: serialize tables to structured text, keep page-level citations so answers are verifiable; document as a known limitation.
- **Numerical reasoning accuracy** — ratio/compare are best-effort; the eval reports them per-category and does not gate success on them.
- **Fiscal-year detection** — the refusal correctness for many `hallucination` items depends on reliable year extraction. Mitigation: extract from cover pages with a fallback, and let the grade node cross-check the asked year against chunk metadata.
- **Local embedding/rerank model download** — first run pulls model weights. Mitigation: document it; models are small (bge-small / bge-reranker-base).

---

## 10. Success criteria
1. Clone → one API key → `make ingest && make run` works end-to-end.
2. `make eval` produces a per-category × per-mode table with a strong **correct-refusal rate** on `hallucination`.
3. Hybrid+rerank measurably beats basic on retrieval metrics; agentic recovers a measurable share of hard queries.
4. Clean module boundaries; `rag/` core unit-tested without a live API.
5. README tells the story a reviewer can grasp in under two minutes.

---

## 11. Course-file triage ledger (match / exceed)
Result of reading every corresponding file in `production-course-main-code` (+ part-6) and comparing scope. "Exceed" = we add production integration the course file (a standalone demo) lacks.

| Concern | Course file | Verdict | What we take / add beyond it |
| --- | --- | --- | --- |
| LLM/embeddings providers | `working_with_llms.py`, `embeddings_deep.py` | **Exceed** | `init_chat_model` provider-swap; add `CacheBackedEmbeddings`, protocol abstraction, config-driven selection |
| Document loading | `document_loaders.py` | **Exceed** | `DirectoryLoader` + `lazy_load`; **PyMuPDF `find_tables()`→Markdown** + bbox subtraction (real table extraction where line-based finds none) |
| Chunking | `text_splitters.py` | **Match+** | Recursive default; expose Markdown-header + token-aware options; carry financial metadata |
| Vector store | `vector_stores.py` | **Exceed** | `similarity_search_with_score` + **filter by `source`** (company/year resolved from `doc_metadata.json`, not denormalized onto chunks); persistent store behind an interface; pgvector seam |
| Advanced retrieval | `advanced_rag.py` | **Match+** | BM25+vector ensemble + compression ideas; add cross-encoder rerank + mode façade |
| Agentic RAG | part-6 `04_agentic_rag.py`, `error_handling.py` | **Match+** | Grade→rewrite→generate loop; **grade node repurposed as a score+metadata refusal gate** |
| Generation / structured output | `rag_pipeline.py`, `output_parsers_final.py` | **Exceed** | `with_structured_output(RAGAnswer)` (answer/citations/refused/computed_value); refusal-first grounded prompt |
| Security | `security_patterns.py` | **Match+** | Full `SecurePipeline`; add structural injection isolation, structured-output guard, provider abstraction, metrics/logging |
| Reliability | `error_handling.py` | **Match** | retry+jitter, circuit breaker, fallback chain |
| Monitoring / logging | `monitoring.py` | **Exceed** | `JSONFormatter` + `MetricsCollector`; add p99 latency; map to `/metrics` + `/health` |
| Caching | `cost_optimization.py` | **Match+** | TTL response cache; semantic-cache + model-routing documented as enhancements |
| Cost / tokens | `cost_optimization.py` | **Match** | `TokenBudget` with `tiktoken`, per-request cap + usage stats |
| Tracing | `langsmith_setup.py` | **Match** | `@traceable` name+tags+metadata; per-request `/chat` trace |
| Testing | `testing_patterns.py` | **Match** | Mock-LLM units + regression harness pattern |
| Evaluation | `testing_patterns.py` | **Exceed** | Local per-category harness **plus** versioned LangSmith dataset + `evaluate()` experiments per mode |
| Contextual retrieval / late chunking / multimodal | part-6 `02/03/06` | **Deferred** | Out of scope; multimodal informs the `pdfplumber` decision; noted as limitations |

