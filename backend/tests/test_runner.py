"""Runner scores predictions with an injected fake answer_fn (no LLM)."""

from eval.golden import GoldenItem
from eval.runner import run_mode
from rag.generation.schema import RAGAnswer


def _items():
    return [
        GoldenItem(question="petra net income 2022?", answer_type="number",
                   answers=[88100000], category="retrieval"),
        GoldenItem(question="crossfirst liabilities 2023?", answer_type="number",
                   answers=["N/A"], category="hallucination"),
    ]


def test_run_mode_scores_each_item():
    def fake_answer(question, mode, deps):
        if "petra" in question:
            return RAGAnswer(answer="88.1", refused=False)
        return RAGAnswer(answer="N/A", refused=True)

    rows = run_mode(_items(), "hybrid", deps=None, answer_fn=fake_answer)
    assert len(rows) == 2
    assert rows[0].correct is True   # 88.1 ≈ 88,100,000
    assert rows[1].correct is True   # refusal matches N/A


def test_run_mode_captures_errors():
    def boom(question, mode, deps):
        raise RuntimeError("provider down")

    rows = run_mode(_items()[:1], "hybrid", deps=None, answer_fn=boom)
    assert rows[0].correct is False
    assert "provider down" in rows[0].error
