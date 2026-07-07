"""Grounded generation returns a structured RAGAnswer (LLM mocked)."""

from langchain_core.documents import Document

from rag.generation.generator import format_context, generate, refusal
from rag.generation.schema import AnswerDraft, RAGAnswer


def _docs():
    return [
        Document(page_content="Total assets were 5,118,490.",
                 metadata={"source": "enrg.pdf", "page": 12, "chunk_id": "enrg.pdf::p12::c0"}),
    ]


def _llm(draft: AnswerDraft):
    class _Structured:
        def invoke(self, _p): return draft

    class _LLM:
        def with_structured_output(self, _schema): return _Structured()

    return _LLM()


def test_format_context_tags_source_and_page():
    ctx = format_context(_docs())
    assert "enrg.pdf" in ctx and "12" in ctx
    assert "Total assets were 5,118,490." in ctx


def test_generate_attaches_citations_from_docs():
    # The LLM returns only a flat draft; citations come from the retrieved chunks.
    draft = AnswerDraft(answer="Total assets were 5,118,490.", refused=False, confidence="high")
    result = generate("What were total assets?", _docs(), _llm(draft))
    assert isinstance(result, RAGAnswer)
    assert result.refused is False
    assert "5,118,490" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].source == "enrg.pdf"
    assert result.citations[0].page == 12


def test_generate_refused_has_no_citations():
    draft = AnswerDraft(answer="N/A", refused=True, confidence="high")
    result = generate("unanswerable?", _docs(), _llm(draft))
    assert result.refused is True
    assert result.citations == []


def test_generate_trims_context_to_token_budget(monkeypatch):
    # Deterministic, offline token counting: 1 token per character.
    import rag.generation.generator as G
    monkeypatch.setattr(G, "count_tokens", lambda text: len(text))

    docs = [Document(page_content="x" * 100,
                     metadata={"source": "a.pdf", "page": i}) for i in range(10)]
    draft = AnswerDraft(answer="ok", refused=False)
    # A tight budget must force lower-ranked chunks to be dropped.
    result = generate("q", docs, _llm(draft), max_tokens=250)
    assert 0 < len(result.citations) < 10  # trimmed, but never empty


def test_generate_no_trim_when_budget_none():
    docs = [Document(page_content="x" * 100,
                     metadata={"source": "a.pdf", "page": i}) for i in range(5)]
    draft = AnswerDraft(answer="ok", refused=False)
    result = generate("q", docs, _llm(draft))  # max_tokens=None -> no trimming
    assert len(result.citations) == 5


def test_refusal_is_grounded_na_without_llm():
    r = refusal("company not in the provided filings")
    assert r.refused is True
    assert r.answer.strip().upper().startswith("N/A")
    assert r.citations == []
