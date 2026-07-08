"""Structured answer types for the RAG generator (framework-free — the API
layer maps these to its own response models)."""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str
    company: str = ""
    fiscal_year: str = ""
    page: int = 0


class AnswerDraft(BaseModel):
    """What the LLM produces — a flat schema (small models are unreliable at
    nested structured output). Citations are attached from the retrieved chunks,
    not asked of the model."""

    answer: str = Field(description="The answer, or 'N/A' if unsupported by the context")
    refused: bool = Field(default=False, description="True if the excerpts don't answer it")
    confidence: str = Field(default="medium", description="high, medium, or low")


class RAGAnswer(BaseModel):
    """The final answer returned to callers, with citations derived from the
    chunks that were actually retrieved."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    confidence: str = "medium"
    reason: str = Field(default="", description="Why this was refused (empty when not refused)")
