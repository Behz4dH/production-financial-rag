"""RAGAnswer -> ChatResponse mapping."""

from app.mapping import to_chat_response
from app.models import ChatResponse
from rag.generation.schema import Citation, RAGAnswer


def test_maps_answer_and_citations():
    rag = RAGAnswer(answer="88.1", refused=False, confidence="high",
                    citations=[Citation(source="petra.pdf", company="Petra Inc",
                                        fiscal_year="2022", page=146)])
    resp = to_chat_response(rag, model_used="llama-3.1-8b-instant",
                            mode="hybrid", cached=False, processing_time_ms=12.3)
    assert isinstance(resp, ChatResponse)
    assert resp.response == "88.1"
    assert resp.refused is False
    assert resp.mode == "hybrid"
    assert resp.citations[0].source == "petra.pdf"
    assert resp.citations[0].company == "Petra Inc"
    assert resp.citations[0].fiscal_year == "2022"
    assert resp.citations[0].page == 146
