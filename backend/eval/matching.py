"""Decide whether a prediction matches the golden answer.

Financial answers vary in scale (a report says "88.1" meaning 88.1 million;
the golden value is 88100000), so numeric matching tries a range of scales
within a relative tolerance. A refusal is correct exactly when "N/A" is an
acceptable golden answer.
"""

import re

from eval.golden import GoldenItem
from rag.generation.schema import RAGAnswer

_SCALES = (1, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)
_SUFFIXES = {"inc", "incorporated", "ltd", "limited", "plc", "corp", "corporation",
             "sa", "ag", "llc", "co", "company", "holdings", "group", "the"}


def parse_number(text: str) -> float | None:
    cleaned = text.replace(",", "")
    negative = "(" in cleaned and ")" in cleaned
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    value = float(match.group())
    if negative and value > 0:
        value = -value
    return value


def number_matches(pred: float, golden: float, rel_tol: float = 0.02) -> bool:
    denom = abs(golden) if golden != 0 else 1.0
    return any(abs(pred * scale - golden) <= rel_tol * denom for scale in _SCALES)


def _tokens(name: str) -> frozenset[str]:
    toks = re.findall(r"[a-z0-9]+", name.lower())
    return frozenset(t for t in toks if t not in _SUFFIXES)


def name_matches(pred: str, golden: str) -> bool:
    a, b = _tokens(pred), _tokens(golden)
    return bool(a) and bool(b) and (b <= a or a <= b)


def _is_na(value) -> bool:
    return isinstance(value, str) and value.strip().upper() == "N/A"


def is_correct(prediction: RAGAnswer, item: GoldenItem) -> bool:
    na_acceptable = any(_is_na(a) for a in item.answers)
    refused = prediction.refused or _is_na(prediction.answer)

    if refused:
        return na_acceptable

    if item.answer_type == "number":
        pred_num = parse_number(prediction.answer)
        if pred_num is None:
            return False
        return any(not _is_na(g) and number_matches(pred_num, float(g))
                   for g in item.answers if isinstance(g, (int, float)))

    # name schema
    return any(not _is_na(g) and name_matches(prediction.answer, str(g))
               for g in item.answers)
