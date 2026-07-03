"""Loader logic tested offline: text/markdown via temp files, PDF via a fake
PyMuPDF (real ``pymupdf.Rect`` for geometry, faked I/O — no real PDF opened)."""

import pymupdf
from langchain_core.documents import Document

from rag.ingestion import loaders


# --- Fake PyMuPDF objects (only the surface load_pdf touches) ---------------


class _FakeTable:
    def __init__(self, rows, cols, bbox, md):
        self.row_count = rows
        self.col_count = cols
        self.bbox = bbox
        self._md = md

    def to_markdown(self):
        return self._md


class _FakeFinder:
    def __init__(self, tables):
        self.tables = tables


class _FakePage:
    def __init__(self, number, text, blocks, tables):
        self.number = number
        self._text = text
        self._blocks = blocks
        self._tables = tables

    def find_tables(self):
        return _FakeFinder(self._tables)

    def get_text(self, mode="text"):
        return self._blocks if mode == "blocks" else self._text


class _FakeDoc:
    def __init__(self, pages):
        self._pages = pages

    def __iter__(self):
        return iter(self._pages)

    def close(self):
        pass


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


def test_load_pdf_renders_tables_as_markdown_and_keeps_page_metadata(monkeypatch, tmp_path):
    # Page 1: a genuine 3x3 table + narrative. The table's block sits INSIDE the
    # table bbox and must be subtracted from the narrative (no duplication).
    table = _FakeTable(3, 3, (0, 15, 60, 60), "| Net sales | $ | 688,415 |")
    blocks_p1 = [
        (0, 0, 10, 10, "Management discussion follows.", 0, 0),  # outside table
        (0, 20, 50, 55, "Net sales 688,415", 1, 0),  # inside table bbox -> dropped
    ]
    page1 = _FakePage(0, "unused when blocks used", blocks_p1, [table])
    # Page 2: no tables -> narrative comes straight from get_text().
    page2 = _FakePage(1, "Total assets 5,118,490", [], [])

    monkeypatch.setattr(loaders.pymupdf, "open", lambda path: _FakeDoc([page1, page2]))

    pdf = tmp_path / "84749ef5.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")
    docs = loaders.load_pdf(pdf)

    assert len(docs) == 2
    assert docs[0].metadata == {"source": "84749ef5.pdf", "page": 1}
    assert "| Net sales | $ | 688,415 |" in docs[0].page_content   # markdown table
    assert "Management discussion follows." in docs[0].page_content  # narrative kept
    assert "Net sales 688,415" not in docs[0].page_content          # table block subtracted
    assert docs[1].metadata["page"] == 2
    assert "Total assets 5,118,490" in docs[1].page_content


def test_load_pdf_filters_non_genuine_tables(monkeypatch, tmp_path):
    # A 1-column "table" (prose wrongly boxed) must be ignored; page text kept.
    prose_box = _FakeTable(4, 1, (0, 0, 40, 40), "| Dear shareholders |")
    page = _FakePage(0, "Dear shareholders, this year we...", [], [prose_box])
    monkeypatch.setattr(loaders.pymupdf, "open", lambda path: _FakeDoc([page]))

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF stub")
    docs = loaders.load_pdf(pdf)

    assert len(docs) == 1
    # No table markdown emitted, and since no genuine table, narrative = get_text().
    assert "| Dear shareholders |" not in docs[0].page_content
    assert "Dear shareholders, this year we..." in docs[0].page_content


def test_load_directory_yields_supported_files(tmp_path):
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("bravo", encoding="utf-8")
    (tmp_path / "skip.csv").write_text("nope", encoding="utf-8")
    batches = list(loaders.load_directory(tmp_path))
    sources = sorted(d.metadata["source"] for batch in batches for d in batch)
    assert sources == ["a.md", "b.txt"]
