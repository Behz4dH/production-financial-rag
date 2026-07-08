"""Smoke-test the query path on real data (real Groq + the persisted index).

Runs three benchmark-style questions that exercise the three key behaviors:
  - answerable      -> a grounded answer with citations
  - wrong fiscal yr -> N/A (company is in the corpus, but not for that year)
  - unknown company -> N/A (company is not in the corpus)

Usage (from backend/):  uv run python dev/smoke_test.py [basic|hybrid|agentic]
Requires GROQ_API_KEY in .env and a built index (run `make ingest` first).
"""

import sys
from pathlib import Path

# Put backend/ on the path so `import rag...` works when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from rag.query import answer, build_deps  # noqa: E402

QUESTIONS = [
    ("answerable",
     'What was the net income of "Petra Diamonds" in the fiscal year 2022?'),
    ("wrong-year (expect N/A)",
     'What was the total liabilities of "CrossFirst Bank" in the fiscal year 2023?'),
    ("unknown-company (expect N/A)",
     'What was the operating cash flow of "JOURNEY MEDICAL CORPORATION" in the fiscal year 2023?'),
]


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
    print(f"Loading index + reranker (first run downloads the reranker model)...")
    deps = build_deps()

    for label, question in QUESTIONS:
        print("\n" + "=" * 72)
        print(f"[{label}]  mode={mode}")
        print(f"Q: {question}")
        result = answer(question, mode, deps)
        print(f"refused: {result.refused}")
        if result.refused:
            print(f"reason : {result.reason}")
        print(f"answer : {result.answer}")
        for c in result.citations:
            print(f"  cite: {c.source} p{c.page}")


if __name__ == "__main__":
    main()
