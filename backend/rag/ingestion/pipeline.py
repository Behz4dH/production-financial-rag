"""Batch ingestion: load → metadata → chunk → embed/persist + docstore.

Run offline via `python -m rag.ingestion.pipeline` (see `make ingest`).
"""

import json
import sys
from pathlib import Path

from langchain_core.documents import Document

from core.config import Settings, get_settings
from rag.ingestion.chunking import chunk_pages
from rag.ingestion.loaders import load_directory
from rag.ingestion.metadata import load_or_extract
from rag.providers.factory import get_embeddings, get_llm
from rag.retrieval.store import ChromaStore, VectorStore


def _docstore_line(chunk: Document) -> str:
    """One JSONL row for the docstore: the chunk text plus its metadata."""
    row = {"page_content": chunk.page_content, "metadata": chunk.metadata}
    return json.dumps(row, ensure_ascii=False) + "\n"


def ingest(settings: Settings, store: VectorStore, llm) -> dict:
    docstore = Path(settings.docstore_path)
    docstore.parent.mkdir(parents=True, exist_ok=True)

    # Fresh vector store each run, in lock-step with the fresh docstore below —
    # so a changed chunking scheme can't leave orphaned vectors behind.
    store.reset()

    files = 0
    total_chunks = 0
    companies: list[str] = []

    # Fresh docstore each run so it stays in lock-step with the vector store.
    with docstore.open("w", encoding="utf-8") as fh:
        for pages in load_directory(settings.data_dir):
            if not pages:
                continue
            source = pages[0].metadata.get("source", "unknown")
            # Extract entity metadata into doc_metadata.json (the single source
            # of truth, resolved at query time) — NOT copied onto the chunks.
            md = load_or_extract(
                source, pages, llm, settings.doc_metadata_path,
                settings.metadata_extract_pages, settings.metadata_extract_chars,
            )
            chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
            store.add(chunks)
            for chunk in chunks:
                fh.write(_docstore_line(chunk))
            files += 1
            total_chunks += len(chunks)
            companies.append(md.company_name)

    return {"files": files, "chunks": total_chunks, "companies": companies}


def main() -> None:
    # The 10-Ks contain non-cp1252 characters; force UTF-8 before any output.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    settings = get_settings()
    store = ChromaStore(get_embeddings(settings), settings.chroma_dir, settings.collection_name)
    summary = ingest(settings, store, get_llm(settings))
    print(f"Ingested {summary['files']} files -> {summary['chunks']} chunks")
    print(f"Companies: {', '.join(sorted(set(summary['companies'])))}")


if __name__ == "__main__":
    main()
