"""Run one retrieval mode over the benchmark and score each answer."""

from pydantic import BaseModel

from core.reliability import with_retry
from eval.golden import GoldenItem
from eval.matching import is_correct
from rag.query import answer

# Sustained benchmark runs exceed provider per-minute token caps, so 429s are
# steady state: retry those with long backoff; record every other error.
_RATE_LIMIT_RETRIES = 5
_RATE_LIMIT_BASE_DELAY = 20.0  # seconds; per-minute windows need time to reset


def _is_rate_limited(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429


def _alias_groups(deps) -> list[list[str]]:
    """Per-filing name groups from deps.entity_index, so name answers given as
    an alias ("MOL") score against the benchmark's spelling ("MITSUI O.S.K.
    LINES")."""
    by_source: dict[str, list[str]] = {}
    for rec in getattr(deps, "entity_index", None) or []:
        by_source.setdefault(rec["source"], []).append(rec["name"])
    return list(by_source.values())


class EvalRow(BaseModel):
    question: str
    category: str
    answer_type: str
    expected: list
    predicted: str
    refused: bool
    correct: bool
    error: str = ""


def run_mode(items: list[GoldenItem], mode: str, deps, answer_fn=answer,
             retry_base_delay: float = _RATE_LIMIT_BASE_DELAY) -> list[EvalRow]:
    rows: list[EvalRow] = []
    aliases = _alias_groups(deps)
    for item in items:
        try:
            prediction = with_retry(
                lambda item=item: answer_fn(item.question, mode, deps),
                max_retries=_RATE_LIMIT_RETRIES, base_delay=retry_base_delay,
                max_delay=60.0, retry_if=_is_rate_limited)
            rows.append(EvalRow(
                question=item.question, category=item.category, answer_type=item.answer_type,
                expected=item.answers, predicted=prediction.answer,
                refused=prediction.refused,
                correct=is_correct(prediction, item, alias_groups=aliases),
            ))
        except Exception as exc:  # noqa: BLE001 — one bad question shouldn't abort the run
            rows.append(EvalRow(
                question=item.question, category=item.category, answer_type=item.answer_type,
                expected=item.answers, predicted="", refused=False,
                correct=False, error=f"{type(exc).__name__}: {exc}",
            ))
    return rows
