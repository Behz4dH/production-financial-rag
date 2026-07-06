"""Load the persisted chunk docstore (docstore.jsonl) into Documents."""

import json
from pathlib import Path

from langchain_core.documents import Document


def load_docstore(path: str) -> list[Document]:
    p = Path(path)
    if not p.exists():
        return []
    docs: list[Document] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        docs.append(Document(page_content=row["page_content"], metadata=row["metadata"]))
    return docs
