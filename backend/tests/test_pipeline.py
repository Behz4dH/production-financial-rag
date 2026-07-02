"""Pipeline wiring tested with fakes: fake embeddings/LLM, temp dirs, tiny docs."""

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from core.config import Settings
from rag.ingestion import pipeline
from rag.ingestion.metadata import DocumentMetadata
from rag.retrieval.store import ChromaStore


class _HashEmbeddings(Embeddings):
    def _vec(self, t):
        v = [0.0] * 8
        for tok in t.lower().split():
            v[hash(tok) % 8] += 1.0
        return v
    def embed_documents(self, texts): return [self._vec(t) for t in texts]
    def embed_query(self, text): return self._vec(text)


class _FakeStructured:
    def __init__(self, name): self._name = name
    def invoke(self, _p):
        return DocumentMetadata(company_name=self._name, fiscal_year="2022",
                                reporting_currency="USD", report_type="10-K")


class _FakeLLM:
    def with_structured_output(self, _schema):
        # Company name derived from nothing here; fixed for determinism.
        return _FakeStructured("TestCorp")


def _settings(tmp_path) -> Settings:
    return Settings(
        groq_api_key="test-key", _env_file=None,
        data_dir=str(tmp_path / "docs"),
        chroma_dir=str(tmp_path / "chroma"),
        docstore_path=str(tmp_path / "docstore.jsonl"),
        doc_metadata_path=str(tmp_path / "doc_metadata.json"),
        chunk_size=200, chunk_overlap=20, metadata_extract_pages=2,
    )


def test_ingest_populates_store_and_docstore(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("Total assets " * 60, encoding="utf-8")
    (docs_dir / "b.md").write_text("Net income " * 60, encoding="utf-8")

    settings = _settings(tmp_path)
    store = ChromaStore(_HashEmbeddings(), settings.chroma_dir, settings.collection_name)

    summary = pipeline.ingest(settings, store, _FakeLLM())

    assert summary["files"] == 2
    assert summary["chunks"] > 2
    assert store.count() == summary["chunks"]

    # Docstore JSONL has one line per chunk, each with entity metadata.
    lines = Path(settings.docstore_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == summary["chunks"]
    first = json.loads(lines[0])
    assert first["metadata"]["company"] == "TestCorp"
    assert "page_content" in first

    # doc_metadata.json cached for both sources.
    md_cache = json.loads(Path(settings.doc_metadata_path).read_text(encoding="utf-8"))
    assert set(md_cache.keys()) == {"a.txt", "b.md"}


def test_reingest_is_idempotent(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("Total assets " * 60, encoding="utf-8")
    settings = _settings(tmp_path)
    store = ChromaStore(_HashEmbeddings(), settings.chroma_dir, settings.collection_name)

    s1 = pipeline.ingest(settings, store, _FakeLLM())
    count_after_first = store.count()
    lines_after_first = Path(settings.docstore_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines_after_first) == s1["chunks"] == count_after_first

    # Re-ingesting the same corpus must NOT duplicate vectors (deterministic
    # chunk_id upserts) and must rewrite the docstore fresh (not append).
    s2 = pipeline.ingest(settings, store, _FakeLLM())
    assert store.count() == count_after_first
    lines_after_second = Path(settings.docstore_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines_after_second) == s2["chunks"] == count_after_first
