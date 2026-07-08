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


def parse_numbers(text: str) -> list[float]:
    cleaned = text.replace(",", "")
    values = []
    for m in re.finditer(r"\((-?\d+(?:\.\d+)?)\)|(-?\d+(?:\.\d+)?)", cleaned):
        if m.group(1) is not None:
            values.append(-abs(float(m.group(1))))
        else:
            values.append(float(m.group(2)))
    return values


def parse_number(text: str) -> float | None:
    values = parse_numbers(text)
    return values[0] if values else None


def number_matches(pred: float, golden: float, rel_tol: float = 0.02) -> bool:
    if golden == 0:
        return abs(pred) <= rel_tol
    denom = abs(golden)
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
        pred_nums = parse_numbers(prediction.answer)
        if not pred_nums:
            return False
        golden_nums = []
        for g in item.answers:
            if _is_na(g):
                continue
            try:
                golden_nums.append(float(g))
            except (TypeError, ValueError):
                continue
        return any(number_matches(p, g) for p in pred_nums for g in golden_nums)

    # name schema
    return any(not _is_na(g) and name_matches(prediction.answer, str(g))
               for g in item.answers)
