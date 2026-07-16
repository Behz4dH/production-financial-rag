"""Decide whether a prediction matches the golden answer.

Financial answers vary in scale ("88.1 million" vs golden 88100000), so
numeric matching tries a range of scales within a relative tolerance — but
inflating a prediction (×10³/10⁶/10⁹) is accepted only when its text states
the scale ("thousand", "$6.74B"): a bare mismatched number is a wrong answer,
not a formatting difference. A refusal is correct exactly when "N/A" is an
acceptable golden answer.
"""

import re

from eval.golden import GoldenItem
from rag.generation.schema import RAGAnswer
from rag.retrieval.entity_resolver import name_matches

_SCALES = (1, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)
_SCALE_CUE = re.compile(r"\b(?:thousand|million|billion|bn|mn|mm)s?\b|\d\s?[kmb]\b",
                        re.IGNORECASE)


def has_scale_cue(text: str) -> bool:
    """True when the text states a scale that could justify inflating its
    numbers ("688,415 thousand", "$6.74B")."""
    return bool(_SCALE_CUE.search(text))


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


def number_matches(pred: float, golden: float, rel_tol: float = 0.02,
                   allow_upscale: bool = True) -> bool:
    """allow_upscale=False forbids inflating pred to meet a larger golden
    (the unstated-units case); deflating a fully-written-out pred against an
    abbreviated golden is always allowed."""
    if golden == 0:
        return abs(pred) <= rel_tol
    denom = abs(golden)
    scales = _SCALES if allow_upscale else tuple(s for s in _SCALES if s <= 1)
    return any(abs(pred * scale - golden) <= rel_tol * denom for scale in scales)


def _is_na(value) -> bool:
    return isinstance(value, str) and value.strip().upper() == "N/A"


def _alias_bridged(pred: str, golden: str, alias_groups: list[list[str]] | None) -> bool:
    """True when pred and golden are both names of the same entity.

    "MOL Group" shares no tokens with "MITSUI O.S.K. LINES", but both match
    names inside the same doc_metadata alias group — without this bridge the
    eval marks a substantively correct answer wrong whenever the model replies
    with a filer's alias instead of the benchmark's chosen spelling.
    """
    for group in alias_groups or []:
        if (any(name_matches(pred, name) for name in group)
                and any(name_matches(golden, name) for name in group)):
            return True
    return False


def is_correct(prediction: RAGAnswer, item: GoldenItem,
               alias_groups: list[list[str]] | None = None) -> bool:
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
        upscale_ok = has_scale_cue(prediction.answer)
        return any(number_matches(p, g, allow_upscale=upscale_ok)
                   for p in pred_nums for g in golden_nums)

    # name schema
    return any(not _is_na(g)
               and (name_matches(prediction.answer, str(g))
                    or _alias_bridged(prediction.answer, str(g), alias_groups))
               for g in item.answers)
