"""Metadata extraction + JSON caching, tested with a fake LLM (no network)."""

import json

from langchain_core.documents import Document

from rag.ingestion import metadata as M
from rag.ingestion.metadata import DocumentMetadata


class _FakeStructuredLLM:
    def __init__(self, result): self._result = result
    def invoke(self, _prompt): return self._result


class _FakeLLM:
    """Records how many times structured extraction is invoked."""

    def __init__(self, result):
        self._result = result
        self.calls = 0

    def with_structured_output(self, _schema):
        self.calls += 1
        return _FakeStructuredLLM(self._result)


def _md():
    return DocumentMetadata(company_name="TransUnion", aliases=["TransUnion LLC"],
                            ticker="TRU", fiscal_year="2022",
                            period_end_date="2022-12-31", reporting_currency="USD",
                            report_type="10-K")


def test_first_pages_text_concatenates_n_pages():
    pages = [Document(page_content=f"page {i}", metadata={"page": i}) for i in range(1, 6)]
    text = M.first_pages_text(pages, 3)
    assert "page 1" in text and "page 3" in text and "page 4" not in text


def test_first_pages_text_truncates_to_max_chars():
    pages = [Document(page_content="x" * 10000, metadata={"page": 1})]
    text = M.first_pages_text(pages, 1, max_chars=5000)
    assert len(text) == 5000


def test_first_pages_text_no_truncation_when_under_cap():
    pages = [Document(page_content="short", metadata={"page": 1})]
    assert M.first_pages_text(pages, 1, max_chars=5000) == "short"


def test_extract_metadata_calls_structured_output():
    llm = _FakeLLM(_md())
    md = M.extract_metadata("TransUnion 2022 annual report ...", llm)
    assert md.company_name == "TransUnion"
    assert md.reporting_currency == "USD"
    assert llm.calls == 1


def test_load_or_extract_writes_then_reads_cache(tmp_path):
    cache = tmp_path / "doc_metadata.json"
    pages = [Document(page_content="TransUnion 2022", metadata={"page": 1})]
    llm = _FakeLLM(_md())

    first = M.load_or_extract("tru.pdf", pages, llm, str(cache), n_pages=3)
    assert first.company_name == "TransUnion"
    assert llm.calls == 1
    assert cache.exists()
    stored = json.loads(cache.read_text(encoding="utf-8"))
    assert stored["tru.pdf"]["reporting_currency"] == "USD"

    # Second call for the same source uses the cache — LLM not invoked again.
    second = M.load_or_extract("tru.pdf", pages, llm, str(cache), n_pages=3)
    assert second.company_name == "TransUnion"
    assert llm.calls == 1
