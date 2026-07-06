"""Grounded generation returns a structured RAGAnswer (LLM mocked)."""

from langchain_core.documents import Document

from rag.generation.generator import format_context, generate, refusal
from rag.generation.schema import RAGAnswer


def _docs():
    return [
        Document(page_content="Total assets were 5,118,490.",
                 metadata={"source": "enrg.pdf", "page": 12, "chunk_id": "enrg.pdf::p12::c0"}),
    ]


def test_format_context_tags_source_and_page():
    ctx = format_context(_docs())
    assert "enrg.pdf" in ctx and "12" in ctx
    assert "Total assets were 5,118,490." in ctx


def test_generate_returns_structured_answer():
    answer = RAGAnswer(answer="Total assets were 5,118,490.",
                       citations=[], refused=False, confidence="high")

    class _Structured:
        def invoke(self, _p): return answer

    class _LLM:
        def with_structured_output(self, _schema): return _Structured()

    result = generate("What were total assets?", _docs(), _LLM())
    assert isinstance(result, RAGAnswer)
    assert result.refused is False
    assert "5,118,490" in result.answer


def test_refusal_is_grounded_na_without_llm():
    r = refusal("company not in the provided filings")
    assert r.refused is True
    assert r.answer.strip().upper().startswith("N/A")
    assert r.citations == []
