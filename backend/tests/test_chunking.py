"""Chunking preserves page + entity metadata as Chroma-safe scalars."""

from langchain_core.documents import Document

from rag.ingestion.chunking import chunk_pages
from rag.ingestion.metadata import DocumentMetadata


def _md():
    return DocumentMetadata(company_name="Petra Diamonds", aliases=["Petra Diamonds Ltd"],
                            ticker="PDL", fiscal_year="2022",
                            period_end_date="2022-06-30", reporting_currency="GBP",
                            report_type="Annual Report")


def _pages():
    long_text = "Financial statements. " * 80  # forces multiple chunks
    return [
        Document(page_content=long_text, metadata={"source": "petra.pdf", "page": 3}),
        Document(page_content="Short note.", metadata={"source": "petra.pdf", "page": 4}),
    ]


def test_chunks_carry_entity_and_page_metadata():
    chunks = chunk_pages(_pages(), _md(), chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 2  # page 3 split into several
    c = chunks[0]
    assert c.metadata["company"] == "Petra Diamonds"
    assert c.metadata["fiscal_year"] == "2022"
    assert c.metadata["reporting_currency"] == "GBP"
    assert c.metadata["source"] == "petra.pdf"
    assert c.metadata["page"] == 3


def test_chunk_metadata_has_no_list_values():
    chunks = chunk_pages(_pages(), _md(), chunk_size=200, chunk_overlap=20)
    for c in chunks:
        assert "aliases" not in c.metadata
        assert all(not isinstance(v, list) for v in c.metadata.values())


def test_chunk_ids_are_unique_and_traceable():
    chunks = chunk_pages(_pages(), _md(), chunk_size=200, chunk_overlap=20)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0].startswith("petra.pdf::p3::c")
