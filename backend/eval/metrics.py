"""Aggregate scored rows into per-category and overall accuracy."""

from collections import defaultdict

from eval.runner import EvalRow


def _stats(rows: list[EvalRow]) -> dict:
    total = len(rows)
    correct = sum(1 for r in rows if r.correct)
    return {"correct": correct, "total": total,
            "accuracy": correct / total if total else 0.0}


def aggregate(rows: list[EvalRow]) -> dict:
    by_category: dict[str, list[EvalRow]] = defaultdict(list)
    for row in rows:
        by_category[row.category].append(row)
    return {
        "overall": _stats(rows),
        "by_category": {cat: _stats(rs) for cat, rs in by_category.items()},
    }


def refusal_accuracy(rows: list[EvalRow]) -> float:
    hallucination = [r for r in rows if r.category == "hallucination"]
    return _stats(hallucination)["accuracy"]
