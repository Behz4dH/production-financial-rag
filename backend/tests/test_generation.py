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
    draft = AnswerDraft(reasoning="excerpt [1] states total assets directly.",
                        answer="Total assets were 5,118,490.", refused=False, confidence="high")
    result = generate("What were total assets?", _docs(), _llm(draft))
    assert isinstance(result, RAGAnswer)
    assert result.refused is False
    assert "5,118,490" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].source == "enrg.pdf"
    assert result.citations[0].page == 12


def test_generate_refused_has_no_citations():
    draft = AnswerDraft(reasoning="no excerpt covers this.", answer="N/A", refused=True, confidence="high")
    result = generate("unanswerable?", _docs(), _llm(draft))
    assert result.refused is True
    assert result.citations == []


def test_generate_trims_context_to_token_budget(monkeypatch):
    # Deterministic, offline token counting: 1 token per character.
    import rag.generation.generator as G
    monkeypatch.setattr(G, "count_tokens_with_margin", lambda text: len(text))

    docs = [Document(page_content="x" * 100,
                     metadata={"source": "a.pdf", "page": i}) for i in range(10)]
    draft = AnswerDraft(reasoning="matched.", answer="ok", refused=False)
    # A tight budget must force lower-ranked chunks to be dropped.
    result = generate("q", docs, _llm(draft), max_tokens=250)
    assert 0 < len(result.citations) < 10  # trimmed, but never empty


def test_generate_no_trim_when_budget_none():
    docs = [Document(page_content="x" * 100,
                     metadata={"source": "a.pdf", "page": i}) for i in range(5)]
    draft = AnswerDraft(reasoning="matched.", answer="ok", refused=False)
    result = generate("q", docs, _llm(draft))  # max_tokens=None -> no trimming
    assert len(result.citations) == 5


def test_refusal_is_grounded_na_without_llm():
    r = refusal("company not in the provided filings")
    assert r.refused is True
    assert r.answer.strip().upper().startswith("N/A")
    assert r.citations == []


def test_fit_context_applies_token_margin():
    """The budget comparison uses the inflated estimate, so chunks that fit
    the raw count but not the margined count get dropped."""
    from langchain_core.documents import Document

    from core.token_budget import count_tokens
    from rag.generation.generator import _fit_context

    docs = [Document(page_content="alpha " * 200, metadata={"source": "a", "page": i})
            for i in range(4)]
    question = "q?"
    # Budget exactly equal to the RAW count of the full assembly: without a
    # margin everything fits; with the 1.25 margin the tail must be dropped.
    from rag.generation.generator import format_context
    raw_total = count_tokens(question + format_context(docs))
    kept = _fit_context(question, docs, raw_total)
    assert len(kept) < len(docs)
    assert kept[0] is docs[0]  # best-first head always survives


def test_fit_context_keeps_every_source_when_multi_source(monkeypatch):
    """Source-blind tail-trimming dropped ALL of one company's pages on
    compare questions — generation then correctly-but-needlessly refused
    ('no information for company B'). Multi-source trimming must keep at
    least one doc per source."""
    import rag.generation.generator as G
    monkeypatch.setattr(G, "count_tokens_with_margin", lambda text: len(text))
    from rag.generation.generator import _fit_context

    docs = ([Document(page_content="a" * 300, metadata={"source": "holley.pdf", "page": i})
             for i in range(3)]
            + [Document(page_content="b" * 300, metadata={"source": "sg.pdf", "page": 9})])
    kept = _fit_context("q", docs, max_tokens=700, multi_source=True)
    sources = {d.metadata["source"] for d in kept}
    assert sources == {"holley.pdf", "sg.pdf"}  # B's last doc survived the trim

    # single-source behavior unchanged: plain tail-trim
    kept_single = _fit_context("q", docs[:3], max_tokens=700)
    assert len(kept_single) == 2


def test_format_context_prefers_company_name_over_source_filename():
    """Sources are opaque hash filenames — on compare questions the model
    can't map '194000c9...pdf' to 'Holley Inc.', so it refuses with 'no
    information for company B' while B's figures sit in context."""
    from rag.generation.generator import format_context

    docs = [Document(page_content="Total sales 688,415",
                     metadata={"source": "194000c9.pdf", "page": 77, "company": "Holley Inc."}),
            Document(page_content="Revenue 24,393,946",
                     metadata={"source": "f06d7ecc.pdf", "page": 47})]
    ctx = format_context(docs)
    assert "(Holley Inc. p77)" in ctx        # company name when stamped
    assert "(f06d7ecc.pdf p47)" in ctx       # filename fallback unchanged
