"""Split pages into chunks, stamping each with Chroma-safe scalar metadata."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.ingestion.metadata import DocumentMetadata


def _entity_fields(md: DocumentMetadata) -> dict:
    """Scalar-only entity metadata (no lists — Chroma rejects them)."""
    return {
        "company": md.company_name,
        "fiscal_year": md.fiscal_year,
        "period_end_date": md.period_end_date or "",
        "reporting_currency": md.reporting_currency or "",
        "ticker": md.ticker or "",
        "report_type": md.report_type or "",
    }


def chunk_pages(
    pages: list[Document],
    metadata: DocumentMetadata,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    entity = _entity_fields(metadata)
    out: list[Document] = []
    for page in pages:
        source = page.metadata.get("source", "unknown")
        page_no = page.metadata.get("page", 0)
        for idx, piece in enumerate(splitter.split_text(page.page_content)):
            if not piece.strip():
                continue
            meta = {
                "source": source,
                "page": page_no,
                "chunk_id": f"{source}::p{page_no}::c{idx}",
                **entity,
            }
            out.append(Document(page_content=piece, metadata=meta))
    return out
