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
