"""Aggregate eval rows into per-category accuracy + headline refusal rate."""

from eval.metrics import aggregate, refusal_accuracy
from eval.runner import EvalRow


def _rows():
    return [
        EvalRow(question="a", category="retrieval", answer_type="number", expected=[1],
                predicted="1", refused=False, correct=True),
        EvalRow(question="b", category="hallucination", answer_type="number", expected=["N/A"],
                predicted="N/A", refused=True, correct=True),
        EvalRow(question="c", category="hallucination", answer_type="number", expected=["N/A"],
                predicted="5", refused=False, correct=False),
    ]


def test_aggregate_overall_and_by_category():
    agg = aggregate(_rows())
    assert agg["overall"] == {"correct": 2, "total": 3, "accuracy": 2 / 3}
    assert agg["by_category"]["hallucination"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert agg["by_category"]["retrieval"]["accuracy"] == 1.0


def test_refusal_accuracy_is_hallucination_only():
    assert refusal_accuracy(_rows()) == 0.5
