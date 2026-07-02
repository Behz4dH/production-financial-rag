# Plan 1 — Foundation & Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the backend project scaffold plus the cross-cutting foundation (validated settings, response cache, structured JSON logging, and typed API models) that every later plan builds on.

**Architecture:** A `uv`-managed Python package under `backend/` following the layered structure from the design spec. This plan builds only the `core/` cross-cutting pieces that have no ML dependencies plus the `app/models.py` request/response contracts, so the whole thing is fast to install and fully unit-tested without any API key.

**Tech Stack:** Python 3.13, `uv`, `pydantic` v2, `pydantic-settings`, `pytest`. (Heavier ML/web deps are declared in `pyproject.toml` now so later plans don't have to edit it, but this plan's code imports none of them.)

## Global Constraints

- Python `requires-python = ">=3.13"`.
- All environment access goes through `core/config.py` — no other module calls `os.getenv`.
- Settings use `pydantic-settings` `BaseSettings` with `@lru_cache` singleton `get_settings()`, `env_file=".env"`, `extra="ignore"`.
- Git commit messages: no AI co-authorship / attribution trailers of any kind.
- Tests must run with **no API key set** and **no network**.
- Every task ends green (`pytest` passing) and is committed.

---

### Task 1: Project scaffold & dependencies

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/.python-version`
- Create: `backend/.env.example`
- Create: `backend/README.md`
- Create: `backend/app/__init__.py`
- Create: `backend/core/__init__.py`
- Create: `backend/rag/__init__.py`
- Create: `backend/eval/__init__.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: an importable `core` package and a working `pytest` setup that later tasks extend.

- [ ] **Step 1: Create `backend/.python-version`**

```
3.13
```

- [ ] **Step 2: Create `backend/pyproject.toml`**

```toml
[project]
name = "financial-rag"
version = "0.1.0"
description = "Hallucination-resistant Financial-Report RAG over SEC 10-K filings"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "fastapi>=0.138.0",
    "uvicorn>=0.49.0",
    "pydantic>=2.13.0",
    "pydantic-settings>=2.14.0",
    "python-dotenv>=1.2.0",
    "slowapi>=0.1.10",
    "langchain>=1.2.0",
    "langchain-core>=1.3.0",
    "langgraph>=1.2.0",
    "langchain-groq>=0.3.0",
    "langchain-huggingface>=1.2.0",
    "langchain-chroma>=1.1.0",
    "langchain-classic>=1.0.0",
    "chromadb>=1.5.0",
    "sentence-transformers>=5.0.0",
    "rank-bm25>=0.2.2",
    "pypdf>=6.0.0",
    "pdfplumber>=0.11.0",
    "numpy>=2.0.0",
    "langsmith>=0.9.0",
    "tiktoken>=0.12.0",
]

[dependency-groups]
dev = [
    "pytest>=9.0.0",
    "httpx>=0.28.0",
]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

Note: the first `uv sync` downloads ML wheels (torch via sentence-transformers) and is large; this is expected for an interview repo and only happens once.

- [ ] **Step 3: Create `backend/.env.example`**

```
# Required
GROQ_API_KEY=your_groq_key_here

# LLM / embeddings (defaults shown)
LLM_PROVIDER=groq
PRIMARY_MODEL=llama-3.1-8b-instant
FALLBACK_MODEL=llama-3.3-70b-versatile
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base

# Vector store / data
VECTOR_STORE=chroma
CHROMA_DIR=data/chroma
DATA_DIR=data/docs

# Retrieval
RETRIEVAL_MODE=hybrid
TOP_K=20
TOP_N=5
BM25_WEIGHT=0.4
VECTOR_WEIGHT=0.6
REFUSAL_SCORE_THRESHOLD=0.3

# Security / reliability / limits
ENABLE_LLM_GUARD=false
MAX_RETRIES=3
MAX_TOKENS_PER_REQUEST=8000
RATE_LIMIT=20/minute
CACHE_TTL_SECONDS=300

# Observability
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=financial-rag

# App
APP_ENV=development
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:5173
```

- [ ] **Step 4: Create the package `__init__.py` files**

Create each of these as an **empty file**:
- `backend/app/__init__.py`
- `backend/core/__init__.py`
- `backend/rag/__init__.py`
- `backend/eval/__init__.py`
- `backend/tests/__init__.py`

- [ ] **Step 5: Create `backend/README.md`**

```markdown
# Financial-Report RAG

Hallucination-resistant Retrieval-Augmented Generation over SEC 10-K filings,
evaluated on an adversarial benchmark. See `../docs/superpowers/specs/` for the
design and `../docs/superpowers/plans/` for the build plans.

## Quickstart

```bash
cd backend
uv sync
cp .env.example .env      # add your GROQ_API_KEY
pytest                    # run the test suite
```

More commands (ingest, run, eval) arrive in later build plans.
```

- [ ] **Step 6: Create `backend/tests/test_smoke.py`**

```python
"""Smoke test: the package imports and pytest is wired up."""


def test_core_package_imports():
    import core  # noqa: F401

    assert core is not None
```

- [ ] **Step 7: Install and run the smoke test**

Run:
```bash
cd backend
uv sync
uv run pytest tests/test_smoke.py -v
```
Expected: `uv sync` completes; `test_core_package_imports` PASSES.

- [ ] **Step 8: Commit**

```bash
git add backend/
git commit -m "chore: scaffold backend package with uv and pytest"
```

---

### Task 2: Settings & config

**Files:**
- Create: `backend/core/config.py`
- Test: `backend/tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class Settings(BaseSettings)` with fields listed below.
  - `get_settings() -> Settings` (cached singleton).
  - `Settings.is_production -> bool` property.
  - Field names later tasks rely on: `groq_api_key: str`, `primary_model: str`, `fallback_model: str`, `llm_provider: str`, `embedding_provider: str`, `embedding_model: str`, `reranker_model: str`, `vector_store: str`, `chroma_dir: str`, `data_dir: str`, `retrieval_mode: str`, `top_k: int`, `top_n: int`, `bm25_weight: float`, `vector_weight: float`, `refusal_score_threshold: float`, `enable_llm_guard: bool`, `max_retries: int`, `max_tokens_per_request: int`, `rate_limit: str`, `cache_ttl_seconds: int`, `langsmith_tracing: bool`, `langsmith_api_key: str`, `langsmith_project: str`, `app_env: str`, `log_level: str`, `cors_origins: str`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_config.py`:
```python
"""Tests for centralized settings."""

import pytest

from core.config import Settings, get_settings


def _make(**overrides) -> Settings:
    """Build Settings without reading a .env file, with a required key default."""
    base = {"groq_api_key": "test-key", "_env_file": None}
    base.update(overrides)
    return Settings(**base)


def test_required_key_missing_raises():
    with pytest.raises(Exception):
        Settings(_env_file=None)  # no groq_api_key provided anywhere


def test_defaults_are_applied():
    s = _make()
    assert s.primary_model == "llama-3.1-8b-instant"
    assert s.embedding_model == "BAAI/bge-small-en-v1.5"
    assert s.vector_store == "chroma"
    assert s.top_k == 20
    assert s.top_n == 5
    assert s.enable_llm_guard is False
    assert s.rate_limit == "20/minute"


def test_is_production_property():
    assert _make(app_env="development").is_production is False
    assert _make(app_env="production").is_production is True


def test_env_overrides_are_typed(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("ENABLE_LLM_GUARD", "true")
    s = Settings(_env_file=None)
    assert s.groq_api_key == "env-key"
    assert s.top_k == 7 and isinstance(s.top_k, int)
    assert s.enable_llm_guard is True


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.config'`.

- [ ] **Step 3: Write minimal implementation**

Create `backend/core/config.py`:
```python
"""
Centralized configuration.
Uses pydantic-settings for validated environment variables. Nothing else in
the codebase should read environment variables directly.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- LLM / embeddings ---
    groq_api_key: str
    llm_provider: str = "groq"
    primary_model: str = "llama-3.1-8b-instant"
    fallback_model: str = "llama-3.3-70b-versatile"
    embedding_provider: str = "huggingface"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    # --- Vector store / data ---
    vector_store: str = "chroma"
    chroma_dir: str = "data/chroma"
    data_dir: str = "data/docs"

    # --- Retrieval ---
    retrieval_mode: str = "hybrid"
    top_k: int = 20
    top_n: int = 5
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    refusal_score_threshold: float = 0.3

    # --- Security / reliability / limits ---
    enable_llm_guard: bool = False
    max_retries: int = 3
    max_tokens_per_request: int = 8000
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300

    # --- Observability ---
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "financial-rag"

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — loaded once, reused everywhere."""
    return Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_config.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/core/config.py backend/tests/test_config.py
git commit -m "feat: add validated settings via pydantic-settings"
```

---

### Task 3: Response cache

**Files:**
- Create: `backend/core/cache.py`
- Test: `backend/tests/test_cache.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `class ResponseCache` with `__init__(self, ttl_seconds: int = 300)`, `get(query: str) -> str | None`, `set(query: str, response: str) -> None`, and a `stats` property returning `{"hits", "misses", "hit_rate", "cached_entries"}`. Keys are normalized (lowercased, stripped) and hashed.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_cache.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.cache'`.

- [ ] **Step 3: Write minimal implementation**

Create `backend/core/cache.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_cache.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/core/cache.py backend/tests/test_cache.py
git commit -m "feat: add TTL response cache with hit/miss stats"
```

---

### Task 4: Structured JSON logging

**Files:**
- Create: `backend/core/logging.py`
- Test: `backend/tests/test_logging.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class JSONFormatter(logging.Formatter)` emitting one JSON object per record with keys `timestamp, level, message, module, function` plus any `record.extra_data` dict merged in.
  - `get_logger(name: str = "financial_rag", level: str = "INFO") -> logging.Logger` that attaches a single `JSONFormatter` `StreamHandler` (idempotent — does not double-add handlers).

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_logging.py`:
```python
"""Tests for structured JSON logging."""

import json
import logging

from core.logging import JSONFormatter, get_logger


def test_formatter_emits_valid_json_with_core_fields():
    record = logging.LogRecord(
        name="x", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello", args=(), exc_info=None, func="myfunc",
    )
    payload = json.loads(JSONFormatter().format(record))
    assert payload["level"] == "INFO"
    assert payload["message"] == "hello"
    assert payload["function"] == "myfunc"
    assert "timestamp" in payload


def test_formatter_merges_extra_data():
    record = logging.LogRecord(
        name="x", level=logging.INFO, pathname=__file__, lineno=1,
        msg="req", args=(), exc_info=None,
    )
    record.extra_data = {"latency_ms": 12.5, "mode": "hybrid"}
    payload = json.loads(JSONFormatter().format(record))
    assert payload["latency_ms"] == 12.5
    assert payload["mode"] == "hybrid"


def test_get_logger_does_not_duplicate_handlers():
    a = get_logger("dup_test")
    b = get_logger("dup_test")
    assert a is b
    assert len(a.handlers) == 1


def test_get_logger_respects_level():
    logger = get_logger("level_test", level="WARNING")
    assert logger.level == logging.WARNING
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_logging.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.logging'`.

- [ ] **Step 3: Write minimal implementation**

Create `backend/core/logging.py`:
```python
"""Structured JSON logging for production log aggregation."""

import json
import logging
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        extra = getattr(record, "extra_data", None)
        if isinstance(extra, dict):
            log_obj.update(extra)
        return json.dumps(log_obj)


def get_logger(name: str = "financial_rag", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_logging.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/core/logging.py backend/tests/test_logging.py
git commit -m "feat: add structured JSON logging"
```

---

### Task 5: API request/response models

**Files:**
- Create: `backend/app/models.py`
- Test: `backend/tests/test_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Citation(BaseModel)`: `company: str`, `fiscal_year: str`, `page: int`, `source: str`.
  - `ChatRequest(BaseModel)`: `message: str` (1..2000), `thread_id: str = "default"`, `mode: str | None = None`.
  - `ChatResponse(BaseModel)`: `response: str`, `citations: list[Citation] = []`, `refused: bool = False`, `thread_id: str`, `model_used: str`, `mode: str`, `cached: bool = False`, `processing_time_ms: float`, `timestamp: str` (auto UTC iso).
  - `MetricsResponse(BaseModel)`: `total_requests: int`, `total_errors: int`, `error_rate: str`, `avg_latency_ms: float`, `p99_latency_ms: float`, `cache_hit_rate: str`, `total_input_tokens: int`, `total_output_tokens: int`.
  - `HealthResponse(BaseModel)`: `status: str`, `app_env: str`, `index_loaded: bool`.
  - `ErrorResponse(BaseModel)`: `error: str`, `detail: str | None = None`, `request_id: str | None = None`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_models.py`:
```python
"""Tests for API request/response models."""

import pytest
from pydantic import ValidationError

from app.models import ChatRequest, ChatResponse, Citation


def test_chat_request_defaults():
    req = ChatRequest(message="What was TransUnion's total assets in 2023?")
    assert req.thread_id == "default"
    assert req.mode is None


def test_chat_request_rejects_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_chat_request_rejects_too_long_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="x" * 2001)


def test_chat_response_auto_timestamp_and_defaults():
    resp = ChatResponse(
        response="N/A",
        thread_id="default",
        model_used="llama-3.1-8b-instant",
        mode="hybrid",
        processing_time_ms=12.3,
    )
    assert resp.refused is False
    assert resp.cached is False
    assert resp.citations == []
    assert "T" in resp.timestamp  # ISO-8601


def test_citation_shape():
    c = Citation(company="TransUnion", fiscal_year="2022", page=115, source="transunion.pdf")
    assert c.page == 115
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models'`.

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/models.py`:
```python
"""API request and response models (Pydantic v2)."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Citation(BaseModel):
    company: str
    fiscal_year: str
    page: int
    source: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000,
                         description="The user's question")
    thread_id: str = Field(default="default", description="Conversation thread ID")
    mode: str | None = Field(default=None, description="basic | hybrid | agentic")


class ChatResponse(BaseModel):
    response: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    thread_id: str
    model_used: str
    mode: str
    cached: bool = False
    processing_time_ms: float
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class MetricsResponse(BaseModel):
    total_requests: int
    total_errors: int
    error_rate: str
    avg_latency_ms: float
    p99_latency_ms: float
    cache_hit_rate: str
    total_input_tokens: int
    total_output_tokens: int


class HealthResponse(BaseModel):
    status: str
    app_env: str
    index_loaded: bool


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_models.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Run the full suite**

Run: `cd backend && uv run pytest -v`
Expected: all tests from Tasks 1–5 PASS (smoke, config, cache, logging, models).

- [ ] **Step 6: Commit**

```bash
git add backend/app/models.py backend/tests/test_models.py
git commit -m "feat: add typed API request/response models"
```

---

## Self-Review

**Spec coverage (Plan 1 slice):**
- Config with `pydantic-settings` + `@lru_cache` + `is_production` + `.env.example` → Task 2. ✅
- TTL response cache (design §3.5) → Task 3. ✅
- Structured JSON logging via `JSONFormatter` (design §3.5) → Task 4. ✅
- Typed request/response/error models incl. `Citation`, `refused`, p99 in metrics (design §3.6) → Task 5. ✅
- Scaffold / uv / `.env.example` / test harness with no API key (design §7, §6) → Task 1. ✅
- Deferred to later plans by design: providers, ingestion, retrieval, generation, security, reliability, monitoring counters, tracing, token budget, API routes, eval, frontend.

**Placeholder scan:** none — every step has full code or exact commands.

**Type consistency:** `get_settings`/`Settings` field names in Task 2 match the design spec §3 and the `.env.example` in Task 1. `ChatResponse.mode`/`model_used`/`citations` in Task 5 match `Citation` from the same task. `ResponseCache.stats` keys in Task 3 match its test. `MetricsResponse` fields align with the monitoring design (avg + p99) for the later monitoring task to satisfy.
