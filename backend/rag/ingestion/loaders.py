"""Load PDFs (PyMuPDF text + real table detection) and plain text files.

PyMuPDF's ``find_tables()`` detects the borderless, whitespace-aligned tables
in financial statements that line-based extractors miss. Each detected table is
rendered into atomic per-row lines (see ``rag.ingestion.tables``) and carried
on the page as ``metadata['table_rows']`` so ``chunk_pages`` can emit each row
as its own unsplittable chunk. Detected table regions are subtracted from the
narrative text so the same figures are not duplicated.

Interface note: one Document *per page* (metadata ``{source, page,
table_rows}``). The extraction engine is an internal detail; the page-oriented
contract that chunking, metadata extraction, and page-level citations depend on
is unchanged. ``table_rows`` is transient page metadata — it never reaches the
docstore (only chunk Documents are persisted).
"""

from collections.abc import Iterator
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

from rag.ingestion.tables import has_data_rows, render_table_rows

_TEXT_EXT = {".md", ".txt"}


def _is_genuine_table(table) -> bool:
    """A genuine table has >= 2 rows AND >= 2 columns.

    ``find_tables()`` over-detects, boxing prose (e.g. a chairman's letter) as a
    single-column 'table'; this filter rejects those. Only *kept* tables are
    subtracted from the narrative, so anything filtered out still survives as
    text.
    """
    return table.row_count >= 2 and table.col_count >= 2


def _page_tables(page) -> list[tuple["pymupdf.Rect", list[str]]]:
    """(bbox, rendered_rows) for each table kept on the page.

    Default (line-based) detection first. Borderless filings — no drawn table
    lines anywhere, e.g. the TransUnion 10-K — detect nothing that way, so a
    page with no default hits retries with strategy="text". Text strategy
    over-boxes prose, so a fallback table is kept only if it renders at least
    one numeric data row: otherwise its region would be subtracted from the
    narrative with nothing emitted in its place (silent data loss).
    """
    tables = [t for t in page.find_tables().tables if _is_genuine_table(t)]
    if tables:
        return [(pymupdf.Rect(t.bbox), render_table_rows(t.extract())) for t in tables]
    kept = []
    for t in page.find_tables(strategy="text").tables:
        if not _is_genuine_table(t):
            continue
        grid = t.extract()
        if not has_data_rows(grid):  # prose over-boxed as a "table"
            continue
        rows = render_table_rows(grid)
        if rows:
            kept.append((pymupdf.Rect(t.bbox), rows))
    return kept


def load_pdf(path: str | Path) -> list[Document]:
    path = Path(path)
    doc = pymupdf.open(str(path))
    try:
        out: list[Document] = []
        for page in doc:
            detected = _page_tables(page)
            boxes = [box for box, _ in detected]

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

            table_rows: list[str] = [row for _, rows in detected for row in rows]

            out.append(
                Document(
                    page_content=narrative,
                    metadata={
                        "source": path.name,
                        "page": page.number + 1,
                        "table_rows": table_rows,
                    },
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
