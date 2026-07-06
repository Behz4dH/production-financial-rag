"""Structured answer types for the RAG generator (framework-free — the API
layer maps these to its own response models)."""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str
    company: str = ""
    fiscal_year: str = ""
    page: int = 0


class RAGAnswer(BaseModel):
    answer: str = Field(description="The answer, or 'N/A' if unsupported by the context")
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    confidence: str = Field(default="medium", description="high, medium, or low")
