"""Load PDFs (pypdf text + pdfplumber tables best-effort) and text files.

pypdf reliably extracts the narrative text and inline financial figures.
pdfplumber table detection is line-based and often finds nothing in
whitespace-aligned financial statements — it is a best-effort bonus, never
a requirement (the numbers are already in the pypdf text).
"""

from collections.abc import Iterator
from pathlib import Path

import pdfplumber
import pypdf
from langchain_core.documents import Document

_TEXT_EXT = {".md", ".txt"}


def serialize_tables(tables: list) -> str:
    lines: list[str] = []
    for table in tables or []:
        for row in table or []:
            cells = [("" if c is None else str(c).strip()) for c in row]
            if any(cells):
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def load_pdf(path: str | Path) -> list[Document]:
    path = Path(path)
    reader = pypdf.PdfReader(str(path))
    # Best-effort table extraction; tolerate any pdfplumber failure.
    tables_by_page: dict[int, str] = {}
    try:
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                serialized = serialize_tables(page.extract_tables())
                if serialized:
                    tables_by_page[i] = serialized
    except Exception:
        tables_by_page = {}

    docs: list[Document] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        table_text = tables_by_page.get(i, "")
        content = text if not table_text else f"{text}\n\n{table_text}"
        docs.append(
            Document(page_content=content, metadata={"source": path.name, "page": i + 1})
        )
    return docs


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
