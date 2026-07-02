"""ChromaStore over a deterministic fake embedding (no model download, local only)."""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.retrieval.store import ChromaStore


class _HashEmbeddings(Embeddings):
    """Deterministic 8-dim embedding from token hashes — no network."""

    def _vec(self, text: str):
        v = [0.0] * 8
        for tok in text.lower().split():
            v[hash(tok) % 8] += 1.0
        return v

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def _docs():
    return [
        Document(page_content="total assets were five billion dollars",
                 metadata={"company": "TransUnion", "fiscal_year": "2022"}),
        Document(page_content="net income was eighty eight million",
                 metadata={"company": "Petra Diamonds", "fiscal_year": "2022"}),
    ]


def test_add_and_count(tmp_path):
    store = ChromaStore(_HashEmbeddings(), str(tmp_path / "chroma"), "t_count")
    store.add(_docs())
    assert store.count() == 2


def test_similarity_search_returns_documents(tmp_path):
    store = ChromaStore(_HashEmbeddings(), str(tmp_path / "chroma"), "t_search")
    store.add(_docs())
    results = store.similarity_search("total assets", k=1)
    assert len(results) == 1
    assert isinstance(results[0], Document)


def test_similarity_search_with_score_returns_pairs(tmp_path):
    store = ChromaStore(_HashEmbeddings(), str(tmp_path / "chroma"), "t_score")
    store.add(_docs())
    pairs = store.similarity_search_with_score("net income", k=2)
    assert len(pairs) == 2
    doc, score = pairs[0]
    assert isinstance(doc, Document)
    assert isinstance(score, float)


def test_metadata_filter_restricts_results(tmp_path):
    store = ChromaStore(_HashEmbeddings(), str(tmp_path / "chroma"), "t_filter")
    store.add(_docs())
    results = store.similarity_search("anything", k=5, filter={"company": "Petra Diamonds"})
    assert results
    assert all(d.metadata["company"] == "Petra Diamonds" for d in results)
