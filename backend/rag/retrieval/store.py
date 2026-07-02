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

    def add(self, docs: list[Document]) -> None:
        if not docs:
            return
        # Use deterministic chunk_id as the vector id so re-ingesting the same
        # corpus upserts (no duplicates) rather than appending.
        ids = [d.metadata["chunk_id"] for d in docs] if all(
            "chunk_id" in d.metadata for d in docs
        ) else None
        self._store.add_documents(docs, ids=ids)

    def similarity_search(
        self, query: str, k: int = 5, filter: dict | None = None
    ) -> list[Document]:
        return self._store.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self, query: str, k: int = 5, filter: dict | None = None
    ) -> list[tuple[Document, float]]:
        return self._store.similarity_search_with_score(query, k=k, filter=filter)

    def count(self) -> int:
        return self._store._collection.count()
