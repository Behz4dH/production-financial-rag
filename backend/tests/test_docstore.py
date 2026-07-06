"""Load the persisted chunk docstore into Documents (for BM25)."""

import json

from langchain_core.documents import Document

from rag.retrieval.docstore import load_docstore


def test_load_docstore_reads_documents(tmp_path):
    p = tmp_path / "docstore.jsonl"
    rows = [
        {"page_content": "total assets were five billion",
         "metadata": {"source": "a.pdf", "page": 3, "chunk_id": "a.pdf::p3::c0"}},
        {"page_content": "net income eighty eight million",
         "metadata": {"source": "b.pdf", "page": 4, "chunk_id": "b.pdf::p4::c0"}},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    docs = load_docstore(str(p))
    assert len(docs) == 2
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "total assets were five billion"
    assert docs[0].metadata["source"] == "a.pdf"


def test_load_docstore_missing_file_returns_empty(tmp_path):
    assert load_docstore(str(tmp_path / "nope.jsonl")) == []
