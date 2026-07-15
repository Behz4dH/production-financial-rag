"""Load PDFs via one layout-aware markdown parse (pymupdf4llm) + text files.

This replaces the geometric era — `find_tables()` strategy juggling, bbox
subtraction, genuineness and junk heuristics — with a single text stream:
pymupdf4llm renders each page to markdown with tables as pipe-tables (labels,
units, headings intact), so data can neither vanish nor duplicate at a
detection boundary. Two corpus-measured cleanups happen here:

- running headers/footers (any line on >=40% of a document's pages, min 5)
  and bare page-number lines are stripped — boilerplate-led chunks pushed
  figures past the reranker's snippet window and fed BM25 junk tokens;
- markdown table blocks become caption'd atomic row lines (see tables.py),
  and the narrative keeps only prose, markup-stripped for indexing.

Interface note: one Document *per page* (metadata ``{source, page,
table_rows}``) — unchanged from the geometric loader, so chunking, metadata
extraction, and page-level citations are untouched. ``table_rows`` is
transient page metadata; only chunk Documents are persisted.
"""

import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pymupdf4llm
from langchain_core.documents import Document

from rag.ingestion.tables import plain_text, render_markdown_table_rows

_TEXT_EXT = {".md", ".txt"}
_BOILERPLATE_MIN_PAGES = 5
_BOILERPLATE_FRACTION = 0.4
_PAGE_NUMBER_LINE = re.compile(r"[\d\s\-–—]*")


def _norm(line: str) -> str:
    return " ".join(line.split())


def _boilerplate_lines(pages: list[dict]) -> set[str]:
    """Lines repeating on >=40% of a document's pages (min 5) — running
    headers/footers. Table lines are never treated as boilerplate."""
    freq: Counter[str] = Counter()
    for page in pages:
        for line in {_norm(ln) for ln in page["text"].splitlines() if ln.strip()}:
            freq[line] += 1
    threshold = max(_BOILERPLATE_MIN_PAGES, _BOILERPLATE_FRACTION * len(pages))
    return {line for line, count in freq.items()
            if count >= threshold and not line.startswith("|")}


def _split_page(lines: list[str]) -> tuple[str, list[str]]:
    """(plain narrative, caption'd table row lines) for one page's markdown."""
    prose: list[str] = []
    rows: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            table = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                table.append(lines[i])
                i += 1
            # nearest preceding non-empty prose line = the table's heading or
            # intro sentence; it stays in the narrative too (it's a heading)
            caption = next((ln for ln in reversed(prose) if ln.strip()), "")
            rows.extend(render_markdown_table_rows(table, plain_text(caption)))
        else:
            prose.append(lines[i])
            i += 1
    return plain_text("\n".join(prose)), rows


def load_pdf(path: str | Path) -> list[Document]:
    path = Path(path)
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
    boilerplate = _boilerplate_lines(pages)
    out: list[Document] = []
    for page in pages:
        kept = [ln for ln in page["text"].splitlines()
                if _norm(ln) not in boilerplate
                and not _PAGE_NUMBER_LINE.fullmatch(_norm(ln))]
        narrative, table_rows = _split_page(kept)
        out.append(Document(
            page_content=narrative,
            metadata={"source": path.name,
                      "page": page["metadata"]["page_number"],
                      "table_rows": table_rows}))
    return out


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
