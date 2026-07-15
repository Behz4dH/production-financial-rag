"""Persistent vector store behind a small interface.

ChromaStore is the default (local, zero-infra). The VectorStore ABC is the
seam a pgvector/Qdrant adapter would implement for a shared, scaled index.
"""

from abc import ABC, abstractmethod

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStore(ABC):
    @abstractmethod
    def add(self, docs: list[Document]) -> None: ...

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 5, filter: dict | None = None
    ) -> list[Document]: ...

    @abstractmethod
    def similarity_search_with_score(
        self, query: str, k: int = 5, filter: dict | None = None
    ) -> list[tuple[Document, float]]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def reset(self) -> None: ...


def _chunk_ids(docs: list[Document]) -> list[str] | None:
    """Deterministic ids from chunk_id so re-adding the same corpus upserts
    (no duplicates). None if any chunk lacks one — then Chroma assigns ids."""
    if all("chunk_id" in d.metadata for d in docs):
        return [d.metadata["chunk_id"] for d in docs]
    return None


class ChromaStore(VectorStore):
    def __init__(
        self,
        embeddings: Embeddings,
        persist_directory: str,
        collection_name: str = "financial_reports",
    ):
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )

    # Chroma rejects a single add larger than this (max_batch_size), which a
    # table-dense filing's chunk count can exceed — so add in sub-batches.
    _MAX_BATCH = 5000

    def add(self, docs: list[Document]) -> None:
        if not docs:
            return
        ids = _chunk_ids(docs)
        for i in range(0, len(docs), self._MAX_BATCH):
            batch = docs[i:i + self._MAX_BATCH]
            batch_ids = ids[i:i + self._MAX_BATCH] if ids is not None else None
            self._store.add_documents(batch, ids=batch_ids)

    def similarity_search(
        self, query: str, k: int = 5, filter: dict | None = None
    ) -> list[Document]:
        return self._store.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self, query: str, k: int = 5, filter: dict | None = None
    ) -> list[tuple[Document, float]]:
        pairs = self._store.similarity_search_with_score(query, k=k, filter=filter)
        # Chroma returns numpy floats; make the contract plain Python floats.
        return [(doc, float(score)) for doc, score in pairs]

    def as_retriever(self, k: int = 5, filter: dict | None = None):
        """A LangChain retriever over this store, optionally filtered."""
        return self._store.as_retriever(search_kwargs={"k": k, "filter": filter})

    def count(self) -> int:
        # No public count on the LangChain wrapper; read the Chroma collection.
        return self._store._collection.count()

    def reset(self) -> None:
        """Drop all vectors so a fresh ingest can't leave stale ones behind.

        Deterministic-id upsert only overwrites chunks whose id recurs; when a
        document's chunking changes (e.g. a new chunk scheme, or a page now
        yielding fewer prose pieces), the old ids are orphaned. Clearing first
        keeps the vector store in lock-step with a fresh docstore."""
        self._store.reset_collection()
