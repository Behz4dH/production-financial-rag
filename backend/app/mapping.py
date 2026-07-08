"""Map the framework-free RAGAnswer to the API's ChatResponse."""

from app.models import ChatResponse, Citation
from rag.generation.schema import RAGAnswer


def to_chat_response(rag_answer: RAGAnswer, *, thread_id: str, model_used: str,
                     mode: str, cached: bool, processing_time_ms: float) -> ChatResponse:
    citations = [Citation(source=c.source, company=c.company,
                          fiscal_year=c.fiscal_year, page=c.page)
                 for c in rag_answer.citations]
    return ChatResponse(
        response=rag_answer.answer,
        citations=citations,
        refused=rag_answer.refused,
        refusal_reason=rag_answer.reason,
        thread_id=thread_id,
        model_used=model_used,
        mode=mode,
        cached=cached,
        processing_time_ms=round(processing_time_ms, 2),
    )
