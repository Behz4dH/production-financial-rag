"""Load the ERC golden benchmark (answers.json) into typed items."""

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldenItem(BaseModel):
    question: str
    answer_type: str  # "number" | "name"  (not `schema`: shadows BaseModel.schema)
    answers: list = Field(default_factory=list)  # acceptable answers (values or "N/A")
    category: str = "other"


def load_golden(path: str) -> list[GoldenItem]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        GoldenItem(
            question=row["question"],
            answer_type=row.get("schema", "number"),
            answers=row.get("answer", []),
            category=row.get("category", "other"),
        )
        for row in rows
    ]
