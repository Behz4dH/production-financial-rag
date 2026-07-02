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
│   │   ├── security.py           # input sanitization + PII masking (real middleware)
│   │   ├── monitoring.py         # request/latency/token counters + metrics snapshot
│   │   ├── logging.py            # structured JSON logging setup
│   │   ├── retry.py              # exponential backoff + model fallback helper
│   │   └── tracing.py            # LangSmith: per-request tracing with metadata
│   ├── rag/                      # framework-free ML core (independently testable)
│   │   ├── providers/
│   │   │   ├── base.py           # LLMProvider, EmbeddingsProvider protocols
│   │   │   └── factory.py        # build providers from settings (Groq/HF default)
│   │   ├── ingestion/
│   │   │   ├── loaders.py        # pypdf text + pdfplumber tables, by extension
│   │   │   ├── metadata.py       # extract company name + fiscal year per doc
│   │   │   ├── chunking.py       # RecursiveCharacterTextSplitter + metadata
│   │   │   └── pipeline.py       # ingest(data_dir): load→chunk→embed→persist (idempotent)
│   │   ├── retrieval/
│   │   │   ├── store.py          # VectorStore interface + ChromaStore impl
│   │   │   ├── hybrid.py         # BM25 ⊕ vector ensemble (weighted)
│   │   │   ├── reranker.py       # cross-encoder rerank (top_k → top_n)
│   │   │   ├── agentic.py        # LangGraph: retrieve→grade→[rewrite→retry]→generate/refuse
│   │   │   └── retriever.py      # façade; mode = basic | hybrid | agentic
│   │   └── generation/
│   │       ├── prompt.py         # refusal-first grounded prompt templates
│   │       └── generator.py      # format cited context → LLM → answer + citations
│   ├── eval/
│   │   ├── golden.py             # load ERC questions.json/answers.json
│   │   ├── matching.py           # numeric tolerance + N/A + multi-valid-answer matching
│   │   ├── metrics.py            # hit-rate@k, MRR; LLM-judge faithfulness/groundedness
│   │   └── evaluate.py           # run all modes → per-category comparison table
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
- **factory.py** — `get_llm(settings)`, `get_embeddings(settings)`. Default: Groq `llama-3.1-8b-instant` + local `sentence-transformers` `BAAI/bge-small-en-v1.5`. Config selects provider; adding OpenAI is a new adapter, not a core change.
- Depends on: `langchain-groq` / `langchain-huggingface` (or direct SDKs); `core.config`.

### 3.2 `rag/ingestion`
- **loaders.py** — dispatch by extension: PDF → `pypdf` for narrative text + `pdfplumber` for tables (serialized to markdown-ish text so tables survive chunking); `.md`/`.txt` → plain read.
- **metadata.py** — best-effort extraction of **company name** and **fiscal year / reporting period** per document (from the first pages / cover). Attached to every chunk. This is what lets the system correctly refuse "wrong fiscal year" questions.
- **chunking.py** — `RecursiveCharacterTextSplitter` (tuned size/overlap for financial text); preserves `{source, company, fiscal_year, page, chunk_id}`.
- **pipeline.py** — `ingest(data_dir, store)`: load → chunk → embed → persist to Chroma; also builds/persists the BM25 corpus. Idempotent (safe re-run). Prints a summary (files, chunks, dims).
- Runs offline via `python -m rag.ingestion` / `make ingest`.

### 3.3 `rag/retrieval`
- **store.py** — `VectorStore` interface (`add`, `similarity_search`, `persist`, `load`); `ChromaStore` persistent impl. pgvector adapter documented as the swap point.
- **hybrid.py** — weighted ensemble of BM25 (keyword) + vector (semantic) retrievers. Targets the "embedding mismatch" failure mode (financial jargon, exact company names).
- **reranker.py** — cross-encoder (`BAAI/bge-reranker-base`) reorders top_k candidates to top_n. Targets "retrieval noise."
- **agentic.py** — LangGraph `StateGraph`: `retrieve → grade → route`. The **grade node doubles as a refusal gate**: it checks whether retrieved chunks actually support the asked company + fiscal year. Routes to `rewrite→retry` (bounded by `max_retries`), `generate`, or a grounded `refuse` (N/A). Adapted from the course Part 6 agentic pattern, wired to *our* store.
- **retriever.py** — façade exposing `retrieve(query, mode)` where `mode ∈ {basic, hybrid, agentic}`, so the API and eval switch strategies uniformly.

### 3.4 `rag/generation`
- **prompt.py** — refusal-first grounded template: "Answer ONLY from the provided report excerpts. If the company or the requested fiscal year is not present, answer exactly `N/A`. Cite company · fiscal year · page for every figure." Separate template variant for ratio computation (best-effort).
- **generator.py** — formats retrieved chunks with source tags → calls LLM → returns `answer` + structured `citations[]`.

### 3.5 `core`
- **config.py** — single `Settings(BaseSettings)`, `.env`-loaded, `@lru_cache` singleton, `is_production` property, `extra="ignore"`. Nothing outside this module reads env vars.
- **cache.py** — TTL response cache with hit/miss stats (reused from reference, key normalized by query).
- **security.py** — `InputSanitizer` (prompt-injection patterns) + `PIIDetector`. PII is masked on the **input path before the text ever reaches the LLM** (emails, phones, SSNs, credit cards, IPs) and re-checked on output; promoted from the reference's demo file into real request/response middleware.
- **monitoring.py** — counters: total requests, errors, latency (avg/p99), tokens in/out, cache hit-rate; exposed via `/metrics`.
- **logging.py** — **structured JSON logging** (one JSON object per log line, suitable for production log aggregation) honoring `log_level`; includes request id, mode, latency, and status per request.
- **retry.py** — retry helper with **exponential backoff** for provider calls (bounded by `max_retries`), plus primary→fallback model escalation.
- **tracing.py** — LangSmith setup; **every `/chat` request is traced with metadata** (thread_id, retrieval mode, model, cached flag, latency, token counts). No-op when `LANGSMITH_API_KEY` is absent.

### 3.6 `app`
- **main.py** — FastAPI app factory; lifespan loads settings, providers, and the persisted store once; wires CORS for the React dev origin.
- **routes.py** — `POST /chat`, `GET /health`, `GET /metrics`, Swagger at `/`.
- **models.py** — `ChatRequest {message, thread_id, mode?}`, `ChatResponse {response, citations[], model_used, mode, cached, processing_time_ms, timestamp}`, `HealthResponse`, `MetricsResponse`, `ErrorResponse`.
- **deps.py** — DI providers for settings/store/cache/retriever.
- **rate_limit.py** — slowapi limiter (`rate_limit` from config).

### 3.7 `eval`
- **golden.py** — parse `questions.json` + `answers.json` into typed records (`question, schema, answers[], category, comment`).
- **matching.py** — verdict logic: numeric tolerance (relative), `N/A` matching, and multiple-valid-answer support (golden answers are lists).
- **metrics.py** — retrieval: hit-rate@k, MRR (did a chunk from the right company/year get retrieved?). Answer: LLM-as-judge faithfulness/groundedness + correctness vs golden.
- **evaluate.py** — run the benchmark across `basic | hybrid | agentic`, print a **per-category × per-mode** table. Headline: correct-refusal rate on `hallucination`.

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
| Error handling + retries | Exponential backoff + model fallbacks | `core/retry.py` |
| Response caching | In-memory cache for duplicate calls | `core/cache.py` |
| Rate limiting | Per-IP throttling via slowapi | `app/rate_limit.py` |
| Structured logging | JSON logs for production aggregation | `core/logging.py` |
| Metrics collection | Request count, latency, token usage | `core/monitoring.py`, `GET /metrics` |
| Health checks | `/health` endpoint | `app/routes.py` |
| Docker deployment | Optional (`Dockerfile` + compose), not required | repo root (optional) |

---

## 6. Testing strategy
pytest, with providers mocked so the suite runs **without any API key**:
- Unit: chunking (metadata preserved), metadata extraction, cache (reuse reference tests), hybrid fusion ordering, reranker ordering, agentic routing/refusal logic (mocked grader), config validation, eval matching (numeric tolerance, N/A, multi-valid).
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
`fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `python-dotenv`, `slowapi`, `langchain`, `langchain-core`, `langgraph`, `langchain-groq`, `langchain-huggingface`, `langchain-chroma`, `chromadb`, `sentence-transformers`, `rank-bm25`, `pypdf`, `pdfplumber`, `langsmith`, `tiktoken`; dev: `pytest`, `httpx`. Frontend: `vite`, `react`.

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
```
