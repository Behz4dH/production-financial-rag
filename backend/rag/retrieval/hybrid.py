"""Hybrid retrieval: weighted ensemble of BM25 (keyword) + vector (semantic),
restricted to the resolved source filing(s).

Built per query because the source set is small (usually 1-3 filings), so BM25
indexes only those filings' chunks and the vector search filters by source.
"""

from langchain_classic.retrievers import EnsembleRetriever

from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.store import ChromaStore


def _source_filter(sources: list[str]) -> dict:
    return {"source": {"$in": sources}} if len(sources) > 1 else {"source": sources[0]}


def vector_only_retriever(store: ChromaStore, sources: list[str], k: int):
    return store.as_retriever(k=k, filter=_source_filter(sources))


def build_hybrid_retriever(
    store: ChromaStore,
    docstore_docs: list,
    sources: list[str],
    k: int,
    bm25_weight: float,
    vector_weight: float,
) -> EnsembleRetriever:
    subset = [d for d in docstore_docs if d.metadata.get("source") in sources]
    bm25 = BM25Retriever.from_documents(subset)
    bm25.k = k
    vector = vector_only_retriever(store, sources, k)
    return EnsembleRetriever(retrievers=[bm25, vector],
                             weights=[bm25_weight, vector_weight])
