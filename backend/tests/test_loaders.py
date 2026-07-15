"""Loader logic tested offline: text/markdown via temp files, PDF via a fake
PyMuPDF (real ``pymupdf.Rect`` for geometry, faked I/O — no real PDF opened)."""

import pymupdf
from langchain_core.documents import Document

from rag.ingestion import loaders


# --- Fake PyMuPDF objects (only the surface load_pdf touches) ---------------


class _FakeTable:
    def __init__(self, rows, cols, bbox, grid=None):
        self.row_count = rows
        self.col_count = cols
        self.bbox = bbox
        self._grid = grid or []

    def extract(self):
        return self._grid


class _FakeFinder:
    def __init__(self, tables):
        self.tables = tables


class _FakePage:
    def __init__(self, number, text, blocks, tables, text_tables=None):
        self.number = number
        self._text = text
        self._blocks = blocks
        self._tables = tables
        self._text_tables = text_tables or []
        self.text_strategy_calls = 0

    def find_tables(self, strategy=None):
        if strategy == "text":
            self.text_strategy_calls += 1
            return _FakeFinder(self._text_tables)
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


def test_load_pdf_emits_table_rows_and_prose_only_narrative(monkeypatch, tmp_path):
    # Page 1: a genuine 3x3 table + narrative. The table's block sits INSIDE the
    # table bbox and must be subtracted from the narrative (no duplication). The
    # table is rendered into atomic row strings carried on metadata["table_rows"],
    # NOT concatenated into page_content.
    grid = [
        ["", "2022", "2021"],
        ["Net sales", "688,415", "600,000"],
    ]
    table = _FakeTable(3, 3, (0, 15, 60, 60), grid)
    blocks_p1 = [
        (0, 0, 10, 10, "Management discussion follows.", 0, 0),  # outside table
        (0, 20, 50, 55, "Net sales 688,415", 1, 0),  # inside table bbox -> dropped
    ]
    page1 = _FakePage(0, "unused when blocks used", blocks_p1, [table])
    # Page 2: no tables -> narrative comes straight from get_text(), rows empty.
    page2 = _FakePage(1, "Total assets 5,118,490", [], [])

    monkeypatch.setattr(loaders.pymupdf, "open", lambda path: _FakeDoc([page1, page2]))

    pdf = tmp_path / "84749ef5.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")
    docs = loaders.load_pdf(pdf)

    assert len(docs) == 2
    assert docs[0].metadata["source"] == "84749ef5.pdf"
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["table_rows"] == ["Net sales -- 2022: 688,415; 2021: 600,000"]
    assert "Management discussion follows." in docs[0].page_content  # narrative kept
    assert "Net sales 688,415" not in docs[0].page_content          # table block subtracted
    assert "688,415" not in docs[0].page_content                    # figures live in table_rows
    assert docs[1].metadata["page"] == 2
    assert docs[1].metadata["table_rows"] == []
    assert "Total assets 5,118,490" in docs[1].page_content


def test_load_pdf_filters_non_genuine_tables(monkeypatch, tmp_path):
    # A 1-column "table" (prose wrongly boxed) must be ignored; page text kept.
    prose_box = _FakeTable(4, 1, (0, 0, 40, 40))
    page = _FakePage(0, "Dear shareholders, this year we...", [], [prose_box])
    monkeypatch.setattr(loaders.pymupdf, "open", lambda path: _FakeDoc([page]))

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF stub")
    docs = loaders.load_pdf(pdf)

    assert len(docs) == 1
    # No genuine table -> no rows emitted, narrative = get_text().
    assert docs[0].metadata["table_rows"] == []
    assert "Dear shareholders, this year we..." in docs[0].page_content


def test_load_pdf_falls_back_to_text_strategy_for_borderless_tables(monkeypatch, tmp_path):
    # Borderless filings (e.g. the TransUnion 10-K) have NO drawn table lines,
    # so the default strategy detects nothing; strategy="text" must be tried.
    grid = [
        ["", "2021", "2020"],
        ["Intangible assets, gross", "5,679.5", "5,516.0"],
    ]
    text_table = _FakeTable(2, 3, (0, 15, 60, 60), grid)
    blocks = [
        (0, 0, 10, 10, "Note 7 discussion.", 0, 0),          # outside table
        (0, 20, 50, 55, "Intangible assets 5,679.5", 1, 0),  # inside -> subtracted
    ]
    page = _FakePage(0, "unused", blocks, tables=[], text_tables=[text_table])
    monkeypatch.setattr(loaders.pymupdf, "open", lambda path: _FakeDoc([page]))

    pdf = tmp_path / "borderless.pdf"
    pdf.write_bytes(b"%PDF stub")
    docs = loaders.load_pdf(pdf)

    assert docs[0].metadata["table_rows"] == [
        "Intangible assets, gross -- 2021: 5,679.5; 2020: 5,516.0"]
    assert "Note 7 discussion." in docs[0].page_content
    assert "5,679.5" not in docs[0].page_content  # region subtracted from narrative


def test_text_strategy_table_without_data_rows_is_ignored(monkeypatch, tmp_path):
    # strategy="text" over-boxes prose as "tables". A fallback table that
    # renders ZERO numeric data rows must be discarded entirely — otherwise
    # its region is subtracted from the narrative with nothing emitted in its
    # place (silent data loss).
    prose_grid = [
        ["Dear shareholders", "this year"],
        ["we delivered", "strong results"],
    ]
    prose_table = _FakeTable(2, 2, (0, 0, 40, 40), prose_grid)
    page = _FakePage(0, "Dear shareholders, this year we delivered strong results.",
                     [], tables=[], text_tables=[prose_table])
    monkeypatch.setattr(loaders.pymupdf, "open", lambda path: _FakeDoc([page]))

    pdf = tmp_path / "prose.pdf"
    pdf.write_bytes(b"%PDF stub")
    docs = loaders.load_pdf(pdf)

    assert docs[0].metadata["table_rows"] == []
    assert "Dear shareholders" in docs[0].page_content  # narrative untouched


def test_text_strategy_not_consulted_when_default_finds_tables(monkeypatch, tmp_path):
    grid = [["", "2022"], ["Net sales", "688,415"]]
    table = _FakeTable(2, 2, (0, 15, 60, 60), grid)
    page = _FakePage(0, "text", [(0, 0, 10, 10, "Narrative.", 0, 0)], tables=[table])
    monkeypatch.setattr(loaders.pymupdf, "open", lambda path: _FakeDoc([page]))

    pdf = tmp_path / "bordered.pdf"
    pdf.write_bytes(b"%PDF stub")
    loaders.load_pdf(pdf)

    assert page.text_strategy_calls == 0  # primary path byte-for-byte unchanged


def test_load_directory_yields_supported_files(tmp_path):
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("bravo", encoding="utf-8")
    (tmp_path / "skip.csv").write_text("nope", encoding="utf-8")
    batches = list(loaders.load_directory(tmp_path))
    sources = sorted(d.metadata["source"] for batch in batches for d in batch)
    assert sources == ["a.md", "b.txt"]
