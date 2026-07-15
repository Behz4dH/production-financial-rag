"""BM25 keyword retrieval over in-memory Documents.

First-party replacement for the sunset langchain-community BM25Retriever,
with the same behavior: whitespace tokenization, top-k by Okapi BM25.
"""

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr
from rank_bm25 import BM25Okapi


class BM25Retriever(BaseRetriever):
    documents: list[Document]
    k: int = 20

    _index: BM25Okapi = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        if not self.documents:
            raise ValueError("BM25Retriever requires at least one document")
        self._index = BM25Okapi([d.page_content.split() for d in self.documents])

    @classmethod
    def from_documents(cls, documents: list[Document], k: int = 20) -> "BM25Retriever":
        return cls(documents=documents, k=k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self._index.get_top_n(query.split(), self.documents, n=self.k)
