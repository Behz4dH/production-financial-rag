"""ChromaStore over a deterministic fake embedding (no model download, local only)."""

from langchain_core.documents import Document

from rag.retrieval.store import ChromaStore
from tests.fakes import HashEmbeddings


def _docs():
    return [
        Document(page_content="total assets were five billion dollars",
                 metadata={"company": "TransUnion", "fiscal_year": "2022"}),
        Document(page_content="net income was eighty eight million",
                 metadata={"company": "Petra Diamonds", "fiscal_year": "2022"}),
    ]


def test_add_and_count(tmp_path):
    store = ChromaStore(HashEmbeddings(), str(tmp_path / "chroma"), "t_count")
    store.add(_docs())
    assert store.count() == 2


def test_add_splits_batches_over_chroma_limit(tmp_path):
    # A single add larger than Chroma's max batch must be sub-batched, not
    # rejected — a table-dense filing can exceed it. Uses a tiny _MAX_BATCH so
    # the test stays fast while still crossing the boundary.
    store = ChromaStore(HashEmbeddings(), str(tmp_path / "chroma"), "t_batch")
    store._MAX_BATCH = 100
    docs = [Document(page_content=f"row {i}",
                     metadata={"chunk_id": f"s::p1::t{i}"}) for i in range(250)]
    store.add(docs)
    assert store.count() == 250


def test_similarity_search_returns_documents(tmp_path):
    store = ChromaStore(HashEmbeddings(), str(tmp_path / "chroma"), "t_search")
    store.add(_docs())
    results = store.similarity_search("total assets", k=1)
    assert len(results) == 1
    assert isinstance(results[0], Document)



def test_metadata_filter_restricts_results(tmp_path):
    store = ChromaStore(HashEmbeddings(), str(tmp_path / "chroma"), "t_filter")
    store.add(_docs())
    results = store.similarity_search("anything", k=5, filter={"company": "Petra Diamonds"})
    assert results
    assert all(d.metadata["company"] == "Petra Diamonds" for d in results)
