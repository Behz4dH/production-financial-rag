"""Run one retrieval mode over the benchmark and score each answer."""

from pydantic import BaseModel, Field

from eval.golden import GoldenItem
from eval.matching import is_correct
from rag.query import answer


class EvalRow(BaseModel):
    question: str
    category: str
    answer_type: str
    expected: list
    predicted: str
    refused: bool
    correct: bool
    error: str = ""


def run_mode(items: list[GoldenItem], mode: str, deps, answer_fn=answer) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for item in items:
        try:
            prediction = answer_fn(item.question, mode, deps)
            rows.append(EvalRow(
                question=item.question, category=item.category, answer_type=item.answer_type,
                expected=item.answers, predicted=prediction.answer,
                refused=prediction.refused, correct=is_correct(prediction, item),
            ))
        except Exception as exc:  # noqa: BLE001 — one bad question shouldn't abort the run
            rows.append(EvalRow(
                question=item.question, category=item.category, answer_type=item.answer_type,
                expected=item.answers, predicted="", refused=False,
                correct=False, error=f"{type(exc).__name__}: {exc}",
            ))
    return rows
