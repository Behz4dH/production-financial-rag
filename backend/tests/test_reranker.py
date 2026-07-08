"""Cross-encoder reranker ordering (fake model — no download)."""

from langchain_core.documents import Document

from rag.retrieval.reranker import rerank


class _FakeCrossEncoder:
    """Scores by keyword presence so ordering is deterministic and checkable."""
    def predict(self, pairs):
        return [2.0 if "assets" in doc.lower() else 0.1 for _q, doc in pairs]


def _docs():
    return [
        Document(page_content="revenue grew this year", metadata={"page": 1}),
        Document(page_content="total assets were five billion", metadata={"page": 2}),
        Document(page_content="board of directors", metadata={"page": 3}),
    ]


def test_rerank_orders_by_score_and_truncates():
    out = rerank("total assets", _docs(), _FakeCrossEncoder(), top_n=2)
    assert len(out) == 2
    assert out[0].metadata["page"] == 2  # the 'assets' doc ranks first


def test_rerank_stamps_score_on_kept_docs():
    out = rerank("total assets", _docs(), _FakeCrossEncoder(), top_n=2)
    assert out[0].metadata["rerank_score"] == 2.0
    assert out[1].metadata["rerank_score"] == 0.1


def test_rerank_empty_is_noop():
    assert rerank("q", [], _FakeCrossEncoder(), top_n=5) == []
