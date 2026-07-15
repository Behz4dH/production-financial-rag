"""Hybrid retriever = BM25 (source-filtered docs) + Chroma vector (source filter)."""

from langchain_core.documents import Document

from rag.retrieval.hybrid import build_hybrid_retriever, vector_only_retriever
from rag.retrieval.store import ChromaStore
from tests.fakes import HashEmbeddings


def _docs():
    return [
        Document(page_content="cross first total assets five billion",
                 metadata={"source": "cross.pdf", "page": 1, "chunk_id": "cross.pdf::p1::c0"}),
        Document(page_content="transunion revenue growth",
                 metadata={"source": "tru.pdf", "page": 1, "chunk_id": "tru.pdf::p1::c0"}),
    ]


def _store(tmp_path):
    store = ChromaStore(HashEmbeddings(), str(tmp_path / "chroma"), "hybrid_test")
    store.add(_docs())
    return store


def test_hybrid_restricts_to_source(tmp_path):
    store = _store(tmp_path)
    retriever = build_hybrid_retriever(store, _docs(), ["cross.pdf"], k=5,
                                       bm25_weight=0.4, vector_weight=0.6)
    results = retriever.invoke("total assets")
    assert results
    assert all(d.metadata["source"] == "cross.pdf" for d in results)


def test_vector_only_restricts_to_source(tmp_path):
    store = _store(tmp_path)
    retriever = vector_only_retriever(store, ["tru.pdf"], k=5)
    results = retriever.invoke("revenue")
    assert all(d.metadata["source"] == "tru.pdf" for d in results)
