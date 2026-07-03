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


def test_chunks_carry_only_source_page_and_id():
    chunks = chunk_pages(_pages(), chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 2  # page 3 split into several
    c = chunks[0]
    assert set(c.metadata.keys()) == {"source", "page", "chunk_id"}
    assert c.metadata["source"] == "petra.pdf"
    assert c.metadata["page"] == 3


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
