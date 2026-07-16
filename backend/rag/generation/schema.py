"""Structured answer types for the RAG generator (framework-free — the API
layer maps these to its own response models)."""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str
    company: str = ""
    fiscal_year: str = ""
    page: int = 0


class AnswerDraft(BaseModel):
    """The LLM's output. Flat on purpose (nested structured output is
    unreliable), and `reasoning` is declared before `answer` on purpose:
    fields fill in declaration order, so the answer is conditioned on the
    written reasoning. Citations are never asked of the model."""

    reasoning: str = Field(description="Follow the matching procedure from the system "
                           "instructions: define the exact metric/entity/period the question "
                           "asks for, check each excerpt against that definition (not just "
                           "something similar, broader, or related), and name which excerpt(s) "
                           "actually match and why -- or, if refusing, which excerpts were "
                           "near-misses and why each falls short. Never derive or estimate a "
                           "figure that isn't directly stated.")
    answer: str = Field(description="The answer, stated with the unit and scale the source "
                        "gives it (e.g. '$688,415 thousand', '79.4 pence'), or 'N/A' if "
                        "unsupported by the context")
    refused: bool = Field(default=False, description="True if the excerpts don't answer it")
    confidence: str = Field(default="medium", description="high, medium, or low")


class RAGAnswer(BaseModel):
    """The final answer returned to callers, with citations derived from the
    chunks that were actually retrieved."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    confidence: str = "medium"
    reason: str = Field(default="", description="Why this was refused (empty when not refused). "
                        "For generation-level refusals this is the model's own reasoning; for "
                        "entity/relevance-gate refusals (no generation call made) it's a plain "
                        "mechanical reason instead.")
    reasoning: str = Field(default="", description="The model's reasoning for this answer -- "
                           "which excerpt(s) it used and why they match. Empty only when refusal "
                           "happened before generation ran (entity or relevance gate).")
