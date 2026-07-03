"""Load PDFs (PyMuPDF text + real table detection) and plain text files.

PyMuPDF's ``find_tables()`` detects the borderless, whitespace-aligned tables
in financial statements that line-based extractors miss, and renders them as
Markdown so rows/columns survive chunking. Detected table regions are
subtracted from the narrative text so the same figures are not duplicated.

Interface note: one Document *per page* (metadata ``{source, page}``). The
extraction engine is an internal detail; the page-oriented contract that
chunking, metadata extraction, and page-level citations depend on is unchanged.
"""

from collections.abc import Iterator
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

_TEXT_EXT = {".md", ".txt"}


def _is_genuine_table(table) -> bool:
    """A genuine table has >= 2 rows AND >= 2 columns.

    ``find_tables()`` over-detects, boxing prose (e.g. a chairman's letter) as a
    single-column 'table'; this filter rejects those. Only *kept* tables are
    subtracted from the narrative, so anything filtered out still survives as
    text.
    """
    return table.row_count >= 2 and table.col_count >= 2


def load_pdf(path: str | Path) -> list[Document]:
    path = Path(path)
    doc = pymupdf.open(str(path))
    try:
        out: list[Document] = []
        for page in doc:
            tables = [t for t in page.find_tables().tables if _is_genuine_table(t)]
            tables_md = "\n\n".join(t.to_markdown() for t in tables)
            boxes = [pymupdf.Rect(t.bbox) for t in tables]

            if boxes:
                # Narrative = text blocks whose box is not inside any table region.
                blocks = page.get_text("blocks")
                narrative = "\n".join(
                    b[4]
                    for b in blocks
                    if not any(pymupdf.Rect(b[:4]) in bx for bx in boxes)
                )
            else:
                narrative = page.get_text()

            content = narrative if not tables_md else f"{narrative}\n\n{tables_md}"
            out.append(
                Document(
                    page_content=content,
                    metadata={"source": path.name, "page": page.number + 1},
                )
            )
        return out
    finally:
        doc.close()


def load_text(path: str | Path) -> list[Document]:
    path = Path(path)
    content = path.read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=content, metadata={"source": path.name, "page": 1})]


def load_document(path: str | Path) -> list[Document]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext in _TEXT_EXT:
        return load_text(path)
    raise ValueError(f"Unsupported file type: {ext}")


def load_directory(dir_path: str | Path) -> Iterator[list[Document]]:
    dir_path = Path(dir_path)
    supported = {".pdf", *_TEXT_EXT}
    for p in sorted(dir_path.iterdir()):
        if p.is_file() and p.suffix.lower() in supported:
            yield load_document(p)
