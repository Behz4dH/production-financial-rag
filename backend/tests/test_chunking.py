"""Chunking carries only source/page/chunk_id — entity metadata is NOT copied
onto chunks (it lives solely in doc_metadata.json, resolved at query time)."""

from langchain_core.documents import Document

from rag.ingestion.chunking import chunk_pages


def _pages():
    long_text = "Financial statements. " * 80  # forces multiple chunks
    return [
        Document(page_content=long_text, metadata={"source": "petra.pdf", "page": 3}),
        Document(page_content="Short note.", metadata={"source": "petra.pdf", "page": 4}),
    ]


def test_chunks_carry_only_source_page_id_and_type():
    chunks = chunk_pages(_pages(), chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 2  # page 3 split into several
    c = chunks[0]
    assert set(c.metadata.keys()) == {"source", "page", "chunk_id", "chunk_type"}
    assert c.metadata["source"] == "petra.pdf"
    assert c.metadata["page"] == 3
    assert c.metadata["chunk_type"] == "prose"


def test_chunks_do_not_carry_entity_fields():
    # Entity attributes must NOT be denormalized onto chunks.
    chunks = chunk_pages(_pages(), chunk_size=200, chunk_overlap=20)
    for c in chunks:
        for leaked in ("company", "fiscal_year", "reporting_currency", "aliases", "ticker"):
            assert leaked not in c.metadata


def test_chunk_metadata_has_no_list_values():
    chunks = chunk_pages(_pages(), chunk_size=200, chunk_overlap=20)
    for c in chunks:
        assert all(not isinstance(v, list) for v in c.metadata.values())


def test_chunk_ids_are_unique_and_traceable():
    chunks = chunk_pages(_pages(), chunk_size=200, chunk_overlap=20)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0].startswith("petra.pdf::p3::c")


def test_table_rows_become_atomic_unsplit_chunks():
    # A page carrying table_rows yields one chunk per row, verbatim, never
    # split — even when a row is longer than chunk_size.
    long_row = "Total shareholders' equity -- " + "31.12.2022: 146,469; " * 30
    page = Document(
        page_content="Some narrative prose about the balance sheet.",
        metadata={"source": "trad.pdf", "page": 134,
                  "table_rows": ["Revenue -- 2022: 1,000; 2021: 900", long_row]},
    )
    chunks = chunk_pages([page], chunk_size=200, chunk_overlap=20)
    table_chunks = [c for c in chunks if c.metadata["chunk_type"] == "table"]
    assert [c.page_content for c in table_chunks] == [
        "Revenue -- 2022: 1,000; 2021: 900", long_row]
    assert table_chunks[0].metadata["chunk_id"] == "trad.pdf::p134::t0"
    assert table_chunks[1].metadata["chunk_id"] == "trad.pdf::p134::t1"
    # prose chunks still present and tagged
    assert any(c.metadata["chunk_type"] == "prose" for c in chunks)
