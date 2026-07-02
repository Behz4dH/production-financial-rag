"""The ERC corpus ships in the repo so the project runs out of the box."""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def test_twenty_pdfs_present():
    pdfs = list((DATA / "docs").glob("*.pdf"))
    assert len(pdfs) == 20


def test_benchmark_files_present_and_aligned():
    questions = json.loads((DATA / "benchmark" / "questions.json").read_text(encoding="utf-8"))
    answers = json.loads((DATA / "benchmark" / "answers.json").read_text(encoding="utf-8"))
    assert len(questions) == len(answers) == 40
    assert questions[0]["question"] == answers[0]["question"]
