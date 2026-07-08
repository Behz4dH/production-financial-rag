"""Load the golden benchmark (answers.json) into typed items."""

import json

from eval.golden import GoldenItem, load_golden


def test_load_golden_parses_items(tmp_path):
    p = tmp_path / "answers.json"
    p.write_text(json.dumps([
        {"question": "net income of Petra 2022?", "schema": "number",
         "answer": [88100000], "comment": "x", "category": "retrieval"},
        {"question": "liabilities of CrossFirst 2023?", "schema": "number",
         "answer": ["N/A"], "comment": "wrong year", "category": "hallucination"},
        {"question": "no category here", "schema": "name", "answer": ["MITSUI"]},
    ]), encoding="utf-8")

    items = load_golden(str(p))
    assert len(items) == 3
    assert isinstance(items[0], GoldenItem)
    assert items[0].answers == [88100000]
    assert items[0].category == "retrieval"
    assert items[2].category == "other"  # missing category defaults
