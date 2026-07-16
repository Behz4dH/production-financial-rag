"""Report assembly + rendering + JSON writing (fake answer_fn, no LLM)."""

import json

from eval.evaluate import build_report, render_table, write_results
from eval.golden import GoldenItem
from rag.generation.schema import RAGAnswer


def _items():
    return [
        GoldenItem(question="petra 2022?", answer_type="number", answers=[88100000],
                   category="retrieval"),
        GoldenItem(question="crossfirst 2023?", answer_type="number", answers=["N/A"],
                   category="hallucination"),
    ]


def _answer_fn(question, mode, deps):
    if "petra" in question:
        return RAGAnswer(answer="88.1 million", refused=False)
    return RAGAnswer(answer="N/A", refused=True)


def test_build_report_covers_each_mode():
    report = build_report(_items(), ["basic", "hybrid"], deps=None, answer_fn=_answer_fn)
    assert set(report["modes"]) == {"basic", "hybrid"}
    assert report["modes"]["hybrid"]["overall"]["accuracy"] == 1.0
    assert report["refusal_accuracy"]["hybrid"] == 1.0


def test_render_table_mentions_modes_and_categories():
    report = build_report(_items(), ["hybrid"], deps=None, answer_fn=_answer_fn)
    table = render_table(report)
    assert "hybrid" in table
    assert "hallucination" in table


def test_write_results_roundtrips(tmp_path):
    report = build_report(_items(), ["hybrid"], deps=None, answer_fn=_answer_fn)
    out = tmp_path / "eval_results.json"
    write_results(report, str(out))
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["modes"]["hybrid"]["overall"]["total"] == 2
