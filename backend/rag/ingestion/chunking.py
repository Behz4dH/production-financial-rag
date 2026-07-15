"""Split pages into chunks.

Each page yields two kinds of chunk, both tagged with ``chunk_type``:
- ``prose``  — the page narrative, split by the recursive character splitter.
- ``table``  — one atomic chunk per detected table row (from the loader's
  ``metadata['table_rows']``), which BYPASSES the splitter so a table row is
  never sliced away from its labels.

Chunk metadata carries only a key back to the source document (``source``,
``page``, ``chunk_id``, ``chunk_type``) — NOT the entity attributes (company /
fiscal_year / currency). Those live solely in ``doc_metadata.json`` (the single
source of truth) and are resolved at query time, so correcting a company or
currency never requires re-embedding the corpus.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(
    pages: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    out: list[Document] = []
    for page in pages:
        source = page.metadata.get("source", "unknown")
        page_no = page.metadata.get("page", 0)
        for idx, piece in enumerate(splitter.split_text(page.page_content)):
            if not piece.strip():
                continue
            out.append(
                Document(
                    page_content=piece,
                    metadata={
                        "source": source,
                        "page": page_no,
                        "chunk_id": f"{source}::p{page_no}::c{idx}",
                        "chunk_type": "prose",
                    },
                )
            )
        # Table rows are already atomic — one chunk each, never split. Distinct
        # `t` id prefix keeps them from colliding with the `c` prose ids.
        for idx, row_text in enumerate(page.metadata.get("table_rows", [])):
            if not row_text.strip():
                continue
            out.append(
                Document(
                    page_content=row_text,
                    metadata={
                        "source": source,
                        "page": page_no,
                        "chunk_id": f"{source}::p{page_no}::t{idx}",
                        "chunk_type": "table",
                    },
                )
            )
    return out
