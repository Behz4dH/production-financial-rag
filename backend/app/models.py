"""API request and response models (Pydantic v2)."""

from datetime import datetime, timezone
from typing import Literal

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
    mode: Literal["basic", "hybrid"] | None = Field(
        default=None, description="basic | hybrid (defaults to configured mode)")
    top_k: int | None = Field(default=None, ge=1, le=50,
                              description="Override retrieval candidate count (testing only; "
                                          "defaults to the server's configured top_k)")
    top_n: int | None = Field(default=None, ge=1, le=20,
                              description="Override reranked chunk count (testing only; "
                                          "defaults to the server's configured top_n)")


class ChatResponse(BaseModel):
    response: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str = Field(default="", description="Why, when refused=true; empty otherwise")
    reasoning: str = Field(default="", description="The model's reasoning for this answer -- "
                           "which excerpt(s) it used and why they match. Empty only when refusal "
                           "happened before generation ran (entity or relevance gate).")
    thread_id: str
    model_used: str
    mode: str
    cached: bool = False
    processing_time_ms: float
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class TraceStepModel(BaseModel):
    stage: str
    elapsed_ms: float
    data: dict


class TraceResponse(ChatResponse):
    steps: list[TraceStepModel] = Field(default_factory=list)


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
