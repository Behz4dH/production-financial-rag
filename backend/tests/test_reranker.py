"""Reranker ordering — cross-encoder and LLM paths (fakes only, no downloads,
no network)."""

import json

from langchain_core.documents import Document

from rag.retrieval.reranker import LLMReranker, rerank


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


# --- LLM reranker (fake chat model implementing .invoke(prompt).content) ---


class _Msg:
    def __init__(self, content): self.content = content


class _FakeChatLLM:
    """Returns a JSON array scoring the 'assets' excerpt high. Records the last
    prompt so we can assert the excerpts were passed with ids."""
    def __init__(self):
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        # Excerpts are listed as "id: text"; score the one mentioning assets.
        scores = []
        for line in prompt.splitlines():
            if line[:1].isdigit() and ":" in line:
                idx = int(line.split(":", 1)[0])
                score = 0.9 if "assets" in line.lower() else 0.0
                scores.append({"id": idx, "score": score})
        return _Msg(json.dumps(scores))


def test_llm_reranker_predict_scores_in_order():
    llm = _FakeChatLLM()
    reranker = LLMReranker(llm, snippet_chars=200, batch_size=40)
    pairs = [("total assets", d.page_content) for d in _docs()]
    scores = reranker.predict(pairs)
    assert scores == [0.0, 0.9, 0.0]  # only the 'assets' doc scores high


def test_llm_reranker_through_rerank_orders_and_stamps():
    reranker = LLMReranker(_FakeChatLLM(), snippet_chars=200)
    out = rerank("total assets", _docs(), reranker, top_n=2)
    assert out[0].metadata["page"] == 2  # the 'assets' doc ranks first
    assert out[0].metadata["rerank_score"] == 0.9


def test_llm_reranker_batches_across_multiple_calls():
    # batch_size=1 forces one LLM call per excerpt; scores must still line up.
    reranker = LLMReranker(_FakeChatLLM(), snippet_chars=200, batch_size=1)
    scores = reranker.predict([("q", d.page_content) for d in _docs()])
    assert scores == [0.0, 0.9, 0.0]


def test_llm_reranker_empty_is_noop():
    assert LLMReranker(_FakeChatLLM()).predict([]) == []


def test_llm_reranker_tolerates_unparseable_output():
    class _BadLLM:
        def invoke(self, _p): return _Msg("sorry, I cannot help with that")
    scores = LLMReranker(_BadLLM()).predict([("q", "anything")])
    assert scores == [0.0]  # defaults to 0.0, no crash
