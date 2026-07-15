"""Retrieval-only vertical slice: does the chunk that actually contains the
answer survive to the candidate set, and if not, at which stage does it drop
-- BM25 alone, vector alone, the fused hybrid pool, or the rerank/top_n cut?

No generation, no LLM calls except embeddings + the cross-encoder reranker.
This isolates retrieval recall from generation quality, on a handful of
hand-verified ground-truth cases: 4 mined for free from the page numbers
already present in data/benchmark/answers.json's `comment` field, plus the
TransUnion intangible-assets case (known-failing, confirmed by hand this
session) as a control -- if the harness doesn't show that one missing, the
harness itself is broken.

Usage (from backend/):  uv run python dev/smoke_test_retrieval.py
Requires a built index (run `make ingest` first). No GROQ_API_KEY needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from langchain_community.retrievers import BM25Retriever  # noqa: E402

import rag.retrieval.hybrid as hybrid  # noqa: E402
from rag.query import build_deps  # noqa: E402
from rag.retrieval.reranker import rerank  # noqa: E402

# Match on the answer figure's substring rather than a fixed chunk_id: with
# row-level table chunking the answer now lives in a `::t{idx}` table chunk, and
# the substring is stable across chunk-scheme changes.
#
# Cases target the CURRENT 5-filing slice corpus; each figure is hand-verified
# with a page reference (see data/benchmark/answers.json comments).
CASES = [
    {
        "label": "Holley net sales FY2022 (p36/p53)",
        "question": 'What was the net sales of "Holley Inc." in the fiscal year 2022?',
        "source": "194000c9109c6fa628f1fed33b44ae4c2b8365f4.pdf",
        "target": "688,415",
    },
    {
        "label": "First Mid total assets FY2022 (p44 balance sheet)",
        "question": 'What was the total assets of "First Mid Bancshares, Inc." in the fiscal year 2022?',
        "source": "e765cdd472cb47fa74ee6a52700c61aca645bbee.pdf",
        "target": "6,744,215",
    },
    {
        "label": "TransUnion intangible assets FY2021 (p122, historic control)",
        "question": 'What was the intangible assets of "TransUnion" in the fiscal year 2021?',
        "source": "85fb23ba2910de45e27f8f40170c0f3576043916.pdf",
        "target": "3,770.6",
    },
    {
        "label": "Baker Steel NAV per share FY2022 (p65)",
        "question": 'What was the net asset value per share of "Baker Steel Resources Trust" in the fiscal year 2022?',
        "source": "84749ef5c2bbf2a302b6614f31727a95bf29f309.pdf",
        "target": "79.4",
    },
    {
        "label": "Safe & Green revenue FY2022",
        "question": 'What was the revenue of "Safe & Green Holdings Corp." in the fiscal year 2022?',
        "source": "f06d7ecc8072de616a4ea35c74e20199de6b0691.pdf",
        "target": "24,393,946",
    },
]


def _rank_of(target: str, docs: list) -> str:
    for i, d in enumerate(docs):
        if target in d.page_content:
            ctype = d.metadata.get("chunk_type", "?")
            return f"FOUND at #{i + 1} of {len(docs)}  [{ctype}]"
    return f"MISSING (of {len(docs)})"


def main() -> None:
    print("Loading index + reranker...")
    deps = build_deps()

    for case in CASES:
        source, target = case["source"], case["target"]
        subset = [d for d in deps.docstore_docs if d.metadata.get("source") == source]

        print("\n" + "=" * 72)
        print(case["label"])
        print(" ", case["question"])

        bm25 = BM25Retriever.from_documents(subset)
        bm25.k = deps.settings.top_k
        print("  BM25 alone:   ", _rank_of(target, bm25.invoke(case["question"])))

        vector_docs = hybrid.vector_only_retriever(
            deps.store, [source], deps.settings.top_k).invoke(case["question"])
        print("  Vector alone: ", _rank_of(target, vector_docs))

        fused_docs = hybrid.build_hybrid_retriever(
            deps.store, deps.docstore_docs, [source], deps.settings.top_k,
            deps.settings.bm25_weight, deps.settings.vector_weight).invoke(case["question"])
        print("  Fused hybrid: ", _rank_of(target, fused_docs))

        reranked = rerank(case["question"], fused_docs, deps.reranker, deps.settings.top_n)
        print("  After rerank: ", _rank_of(target, reranked))


if __name__ == "__main__":
    main()
