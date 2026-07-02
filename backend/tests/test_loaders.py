"""Loader logic tested offline: text/markdown via temp files, PDF via fakes."""

from pathlib import Path

from langchain_core.documents import Document

from rag.ingestion import loaders


def test_serialize_tables_joins_rows_and_cells():
    tables = [[["Assets", "2022"], ["Total", "5,118,490"]]]
    out = loaders.serialize_tables(tables)
    assert "Assets | 2022" in out
    assert "Total | 5,118,490" in out


def test_serialize_tables_empty_returns_empty_string():
    assert loaders.serialize_tables([]) == ""
    assert loaders.serialize_tables([[]]) == ""


def test_load_text_reads_markdown(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# Heading\n\nBody text.", encoding="utf-8")
    docs = loaders.load_text(p)
    assert len(docs) == 1
    assert docs[0].metadata == {"source": "note.md", "page": 1}
    assert "Body text." in docs[0].page_content


def test_load_document_dispatches_and_rejects_unknown(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b", encoding="utf-8")
    import pytest
    with pytest.raises(ValueError):
        loaders.load_document(p)


def test_load_pdf_merges_text_and_tables_with_page_metadata(monkeypatch, tmp_path):
    # Fake pypdf: two pages of text.
    class _Page:
        def __init__(self, t): self._t = t
        def extract_text(self): return self._t

    class _Reader:
        def __init__(self, path): self.pages = [_Page("Net income 88"), _Page("Total assets 5,118,490")]

    monkeypatch.setattr(loaders.pypdf, "PdfReader", _Reader)

    # Fake pdfplumber: page 1 has a table, page 2 has none.
    class _PlPage:
        def __init__(self, tables): self._tables = tables
        def extract_tables(self): return self._tables

    class _Pdf:
        def __init__(self): self.pages = [_PlPage([[["Rev", "10"]]]), _PlPage([])]
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(loaders.pdfplumber, "open", lambda path: _Pdf())

    pdf = tmp_path / "84749ef5.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")
    docs = loaders.load_pdf(pdf)

    assert len(docs) == 2
    assert docs[0].metadata == {"source": "84749ef5.pdf", "page": 1}
    assert "Net income 88" in docs[0].page_content
    assert "Rev | 10" in docs[0].page_content        # table merged on page 1
    assert docs[1].metadata["page"] == 2
    assert "Total assets 5,118,490" in docs[1].page_content


def test_load_directory_yields_supported_files(tmp_path):
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("bravo", encoding="utf-8")
    (tmp_path / "skip.csv").write_text("nope", encoding="utf-8")
    batches = list(loaders.load_directory(tmp_path))
    sources = sorted(d.metadata["source"] for batch in batches for d in batch)
    assert sources == ["a.md", "b.txt"]
