"""FastAPI application: /health, /metrics, /chat around rag.query.answer."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app.mapping import to_chat_response
from app.models import ChatRequest, ChatResponse, ErrorResponse, HealthResponse, MetricsResponse
from app.rate_limit import limiter
from core.cache import ResponseCache
from core.config import get_settings
from core.monitoring import MetricsCollector
from core.reliability import with_retry
from core.security import mask_output, screen_input
from core.token_budget import count_tokens, within_budget
from core.tracing import configure_tracing
from rag.query import answer, build_deps


def create_app(query_deps=None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_tracing(settings)
        app.state.query_deps = query_deps if query_deps is not None else build_deps(settings)
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
        return HealthResponse(status="ok", app_env=settings.app_env,
                              index_loaded=request.app.state.query_deps is not None)

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics(request: Request) -> MetricsResponse:
        return MetricsResponse(**request.app.state.metrics.get_summary())

    @app.post("/chat", response_model=ChatResponse)
    @limiter.limit(settings.rate_limit)
    def chat(request: Request, body: ChatRequest):
        state = request.app.state
        started = time.perf_counter()
        mode = body.mode or settings.retrieval_mode

        ok, input_tokens = within_budget(body.message, settings.max_tokens_per_request)
        if not ok:
            return JSONResponse(status_code=413,
                                content=ErrorResponse(error="message too large").model_dump())

        blocked, cleaned = screen_input(body.message)
        if blocked:
            return JSONResponse(status_code=400,
                                content=ErrorResponse(error="input rejected by safety filter").model_dump())

        cached_answer = state.cache.get(f"{mode}:{cleaned}")
        if cached_answer is not None:
            elapsed = (time.perf_counter() - started) * 1000
            parsed_cached = ChatResponse.model_validate_json(cached_answer)
            state.metrics.record_request(elapsed, input_tokens, count_tokens(parsed_cached.response),
                                         cached=True)
            return parsed_cached.model_copy(update={"cached": True})

        try:
            rag_answer = with_retry(
                lambda: answer(cleaned, mode, state.query_deps),
                max_retries=settings.max_retries, base_delay=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - started) * 1000
            state.metrics.record_request(elapsed, input_tokens, 0, error=True)
            return JSONResponse(status_code=500,
                                content=ErrorResponse(error="generation failed",
                                                      detail=str(exc)).model_dump())

        rag_answer.answer = mask_output(rag_answer.answer)
        elapsed = (time.perf_counter() - started) * 1000
        resp = to_chat_response(rag_answer, thread_id=body.thread_id,
                                model_used=settings.primary_model, mode=mode,
                                cached=False, processing_time_ms=elapsed)
        state.cache.set(f"{mode}:{cleaned}", resp.model_dump_json())
        state.metrics.record_request(elapsed, input_tokens, count_tokens(resp.response))
        return resp

    return app
