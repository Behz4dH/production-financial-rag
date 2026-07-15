"""The corpus and benchmark ship in the repo so the project runs out of the box.

Counts are intentionally NOT hardcoded: the working corpus is currently a
5-filing slice (the full 20-filing set is parked outside the repo, with the
full benchmark backed up as *.full20.json) so iteration stays inside free-tier
token quotas. These tests pin the invariants that must hold at ANY corpus
size: documents exist, the benchmark is non-empty, and questions.json stays
aligned row-for-row with answers.json.
"""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def test_corpus_pdfs_present():
    pdfs = list((DATA / "docs").glob("*.pdf"))
    assert len(pdfs) >= 1


def test_benchmark_files_present_and_aligned():
    questions = json.loads((DATA / "benchmark" / "questions.json").read_text(encoding="utf-8"))
    answers = json.loads((DATA / "benchmark" / "answers.json").read_text(encoding="utf-8"))
    assert len(questions) == len(answers) >= 1
    assert [q["question"] for q in questions] == [a["question"] for a in answers]


def test_benchmark_has_refusal_and_answerable_items():
    """The benchmark must exercise BOTH failure modes: hallucination items
    that demand refusal, and real-answer items that punish over-refusing."""
    answers = json.loads((DATA / "benchmark" / "answers.json").read_text(encoding="utf-8"))
    def is_na(row):
        return all(isinstance(a, str) and a.strip().upper() == "N/A"
                   for a in row.get("answer", []))
    assert any(is_na(row) for row in answers)
    assert any(not is_na(row) for row in answers)
