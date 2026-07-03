"""Split pages into chunks.

Chunk metadata carries only a key back to the source document (``source``,
``page``, ``chunk_id``) — NOT the entity attributes (company / fiscal_year /
currency). Those live solely in ``doc_metadata.json`` (the single source of
truth) and are resolved at query time, so correcting a company or currency
never requires re-embedding the corpus.
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
                    },
                )
            )
    return out
