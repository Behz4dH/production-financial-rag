"""FastAPI application: /health, /metrics, /chat around rag.query.answer."""

import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app.mapping import to_chat_response
from app.models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    TraceResponse,
    TraceStepModel,
)
from app.rate_limit import limiter
from core.cache import ResponseCache
from core.config import get_settings
from core.logging import get_logger
from core.monitoring import MetricsCollector
from core.reliability import call_with_fallback, is_retryable, with_retry
from core.security import screen_input
from core.token_budget import count_tokens
from core.tracing import configure_tracing
from rag.providers.factory import get_llm
from rag.query import answer, build_deps
from rag.trace import TraceRecorder


def _scoped_deps(deps, top_k: int | None, top_n: int | None):
    """deps with top_k/top_n overridden for one request only.

    Never mutates the shared deps.settings on app.state — that instance is
    reused across every concurrent request, so overriding it in place would
    leak one caller's test override into everyone else's requests.
    """
    overrides = {k: v for k, v in {"top_k": top_k, "top_n": top_n}.items() if v is not None}
    if not overrides:
        return deps
    return replace(deps, settings=deps.settings.model_copy(update=overrides))


def _error_response(exc: Exception, request_id: str, settings) -> JSONResponse:
    """Map failures to meaningful responses instead of a generic 500.

    Provider rate limits become a 429 the client can act on. Everything else
    is a 500 carrying the request_id (so a report can be matched to the log
    line); raw exception detail is a dev-only convenience, never shown in
    production."""
    if getattr(exc, "status_code", None) == 429:
        return JSONResponse(status_code=429,
                            content=ErrorResponse(error="provider rate limited - retry shortly",
                                                  request_id=request_id).model_dump())
    detail = None if settings.is_production else str(exc)
    return JSONResponse(status_code=500,
                        content=ErrorResponse(error="generation failed", detail=detail,
                                              request_id=request_id).model_dump())


def _screen(message: str) -> tuple[str, JSONResponse | None]:
    """(cleaned_message, None) normally, or ("", error_response) if blocked.
    Shared by /chat and /chat/trace so the two can't drift on this check."""
    blocked, cleaned = screen_input(message)
    if blocked:
        return "", JSONResponse(status_code=400,
                                content=ErrorResponse(error="input rejected by safety filter").model_dump())
    return cleaned, None


def _build_response(rag_answer, body: ChatRequest, mode: str, settings, elapsed_ms: float) -> ChatResponse:
    """Assemble the API response shape. Shared by /chat and /chat/trace so the
    two can't drift. PII masking is input-only (see core.security): answers
    derive from public filings, and masking them corrupted plain 10-digit
    figures (share counts) into '[PHONE REDACTED]'."""
    return to_chat_response(rag_answer, thread_id=body.thread_id, model_used=settings.primary_model,
                            mode=mode, cached=False, processing_time_ms=elapsed_ms)


def create_app(query_deps=None) -> FastAPI:
    settings = get_settings()
    logger = get_logger("financial_rag", settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_tracing(settings)
        if query_deps is not None:  # test/injection path
            app.state.query_deps = query_deps
            app.state.query_deps_fallback = query_deps
        else:
            deps = build_deps(settings)
            app.state.query_deps = deps
            # A fallback deps that shares everything but swaps in the fallback model.
            app.state.query_deps_fallback = replace(
                deps, llm=get_llm(settings, settings.fallback_model))
        app.state.cache = ResponseCache(settings.cache_ttl_seconds)
        app.state.metrics = MetricsCollector()
        yield

    app = FastAPI(title="Financial-Report RAG API", lifespan=lifespan)
    app.state.limiter = limiter
    app.add_exception_handler(
        RateLimitExceeded,
        lambda r, e: JSONResponse(status_code=429, content={"error": "rate limit exceeded"}),
    )
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins.split(","),
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        deps = request.app.state.query_deps
        loaded = deps is not None and deps.store.count() > 0  # the index actually has chunks
        return HealthResponse(status="ok", app_env=settings.app_env, index_loaded=loaded)

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics(request: Request) -> MetricsResponse:
        return MetricsResponse(**request.app.state.metrics.get_summary())

    @app.post("/chat", response_model=ChatResponse)
    @limiter.limit(settings.rate_limit)
    def chat(request: Request, body: ChatRequest):
        state = request.app.state
        started = time.perf_counter()
        request_id = uuid.uuid4().hex[:8]
        mode = body.mode or settings.retrieval_mode
        input_tokens = count_tokens(body.message)

        def log(status: str, elapsed_ms: float, **extra) -> None:
            logger.info("chat", extra={"extra_data": {
                "request_id": request_id, "mode": mode,
                "latency_ms": round(elapsed_ms, 2), "status": status, **extra}})

        # The message is already length-capped by ChatRequest; the token budget
        # that matters (the assembled prompt) is enforced in rag.generation.generate.
        cleaned, error = _screen(body.message)
        if error is not None:
            log("blocked", (time.perf_counter() - started) * 1000)
            return error

        # top_k/top_n in the cache key too — otherwise an overridden request could
        # be served (or serve) a cached answer computed under different settings.
        # This is the ONE place cache-key semantics live (ResponseCache does no
        # normalization of its own), so the message is normalized here.
        cache_key = f"{mode}:{body.top_k}:{body.top_n}:{cleaned.strip().lower()}"
        cached_answer = state.cache.get(cache_key)
        if cached_answer is not None:
            elapsed = (time.perf_counter() - started) * 1000
            parsed = ChatResponse.model_validate_json(cached_answer)
            state.metrics.record_request(elapsed, input_tokens, count_tokens(parsed.response), cached=True)
            log("cached", elapsed, refused=parsed.refused, reason=parsed.refusal_reason)
            return parsed.model_copy(update={"cached": True})

        query_deps = _scoped_deps(state.query_deps, body.top_k, body.top_n)
        fallback_deps = _scoped_deps(state.query_deps_fallback, body.top_k, body.top_n)
        try:
            # Retry the primary model with exponential backoff — but only for
            # transient errors (is_retryable); a deterministic 4xx goes straight
            # to the fallback model, whose limits differ. If both fail: 500.
            rag_answer = call_with_fallback(
                lambda: with_retry(lambda: answer(cleaned, mode, query_deps),
                                   max_retries=settings.request_max_retries,
                                   base_delay=settings.retry_base_delay,
                                   retry_if=is_retryable),
                lambda: answer(cleaned, mode, fallback_deps),
            )
        except Exception as exc:  # noqa: BLE001 — primary (with retries) and fallback both failed
            elapsed = (time.perf_counter() - started) * 1000
            state.metrics.record_request(elapsed, input_tokens, 0, error=True)
            log("error", elapsed, error=str(exc))
            return _error_response(exc, request_id, settings)

        elapsed = (time.perf_counter() - started) * 1000
        resp = _build_response(rag_answer, body, mode, settings, elapsed)
        state.cache.set(cache_key, resp.model_dump_json())
        state.metrics.record_request(elapsed, input_tokens, count_tokens(resp.response))
        log("ok", elapsed, refused=resp.refused, reason=resp.refusal_reason)
        return resp

    @app.post("/chat/trace", response_model=TraceResponse)
    @limiter.limit(settings.rate_limit)
    def chat_trace(request: Request, body: ChatRequest):
        state = request.app.state
        started = time.perf_counter()
        request_id = uuid.uuid4().hex[:8]
        mode = body.mode or settings.retrieval_mode

        cleaned, error = _screen(body.message)
        if error is not None:
            return error

        query_deps = _scoped_deps(state.query_deps, body.top_k, body.top_n)
        recorder = TraceRecorder()
        try:
            # Demo/debug endpoint: one real attempt, no retry/fallback/cache -
            # what you see here is exactly what ran, not a hidden second try.
            rag_answer = answer(cleaned, mode, query_deps, trace=recorder)
        except Exception as exc:  # noqa: BLE001
            return _error_response(exc, request_id, settings)

        elapsed = (time.perf_counter() - started) * 1000
        base = _build_response(rag_answer, body, mode, settings, elapsed)
        return TraceResponse(**base.model_dump(),
                             steps=[TraceStepModel(stage=s.stage, elapsed_ms=s.elapsed_ms, data=s.data)
                                    for s in recorder.steps])

    @app.get("/eval-results")
    def eval_results(request: Request):
        path = Path(settings.data_dir).parent / "eval_results.json"
        if not path.exists():
            return JSONResponse(status_code=404,
                                content={"error": "no eval run yet - run `make eval`"})
        return json.loads(path.read_text(encoding="utf-8"))

    return app
