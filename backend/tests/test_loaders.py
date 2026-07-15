"""Loader contract tests, offline: pymupdf4llm is faked via monkeypatch.

The interface is unchanged from the geometric era — one Document per page
with metadata {source, page, table_rows} — but the inside is a single
layout-aware markdown parse: no table detection strategies, no bbox
subtraction, no genuineness/junk heuristics."""

from langchain_core.documents import Document  # noqa: F401  (fixture parity)

from rag.ingestion import loaders


def _page(number: int, text: str) -> dict:
    return {"metadata": {"page_number": number}, "text": text}


def _fake_to_markdown(pages):
    def fake(path, page_chunks=True, show_progress=False):
        return pages
    return fake


def test_load_pdf_emits_captioned_table_rows_and_plain_narrative(monkeypatch, tmp_path):
    md = """Management discussion follows.

# Consolidated Income Statement

|US$ million|**2022**|2021|
|---|---|---|
|Revenue|**585.2**|406.9|

Notes continue here."""
    monkeypatch.setattr(loaders.pymupdf4llm, "to_markdown",
                        _fake_to_markdown([_page(1, md)]))
    pdf = tmp_path / "filing.pdf"
    pdf.write_bytes(b"%PDF stub")

    docs = loaders.load_pdf(pdf)

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "filing.pdf"
    assert docs[0].metadata["page"] == 1
    # the table's data row is an atomic caption'd unit, NOT in the narrative
    assert docs[0].metadata["table_rows"] == [
        "Consolidated Income Statement — Revenue: 2022: 585.2; 2021: 406.9"]
    assert "585.2" not in docs[0].page_content
    # narrative is markup-stripped prose
    assert "Management discussion follows." in docs[0].page_content
    assert "Notes continue here." in docs[0].page_content
    assert "|" not in docs[0].page_content and "#" not in docs[0].page_content


def test_load_pdf_strips_running_headers_and_page_numbers(monkeypatch, tmp_path):
    boiler = "Strategic Report Corporate Governance Financial Statements"
    pages = [_page(i + 1, f"{boiler}\n{i + 1}\nUnique content {i}.") for i in range(12)]
    monkeypatch.setattr(loaders.pymupdf4llm, "to_markdown", _fake_to_markdown(pages))
    pdf = tmp_path / "boiler.pdf"
    pdf.write_bytes(b"%PDF stub")

    docs = loaders.load_pdf(pdf)

    assert all(boiler not in d.page_content for d in docs)      # running header gone
    assert "Unique content 3." in docs[3].page_content          # real content kept
    assert all(d.page_content.strip() != str(i + 1) for i, d in enumerate(docs))


def test_load_pdf_rare_lines_survive_stripping(monkeypatch, tmp_path):
    # a line on 2 of 12 pages is content, not boilerplate
    rare = "Total assets were 5,118,490 thousand."
    pages = [_page(i + 1, rare if i < 2 else f"Filler {i}.") for i in range(12)]
    monkeypatch.setattr(loaders.pymupdf4llm, "to_markdown", _fake_to_markdown(pages))
    pdf = tmp_path / "rare.pdf"
    pdf.write_bytes(b"%PDF stub")

    docs = loaders.load_pdf(pdf)
    assert rare in docs[0].page_content


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


def test_load_directory_yields_supported_files(tmp_path):
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("bravo", encoding="utf-8")
    (tmp_path / "skip.csv").write_text("nope", encoding="utf-8")
    batches = list(loaders.load_directory(tmp_path))
    sources = sorted(d.metadata["source"] for batch in batches for d in batch)
    assert sources == ["a.md", "b.txt"]
