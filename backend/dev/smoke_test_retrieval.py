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
CASES = [
    {
        "label": "MITSUI shareholders' equity FY2022",
        "question": "What was the shareholders' equity of Mitsui O.S.K. Lines in fiscal year 2022?",
        "source": "6054ec55767fbe6585598ced7afacf5cb8619a13.pdf",
        "target": "1,274,570",
    },
    {
        "label": "Tradition shareholders' equity FY2022",
        "question": "What was the total shareholders' equity of Compagnie Financiere Tradition SA in fiscal year 2022?",
        "source": "2779336b845a41544348abb7b3e6e5bd2ff893a2.pdf",
        "target": "146,469",
    },
    {
        "label": "Petra Diamonds net income FY2022",
        "question": "What was the net income of Petra Diamonds in fiscal year 2022?",
        "source": "609042c64a759c0ac63e7cf18742be4dd3cc5cd5.pdf",
        "target": "88.1",
    },
    {
        "label": "Sensata free cash flow FY2021",
        "question": "What was the free cash flow of Sensata in fiscal year 2021?",
        "source": "e33544bdea57faa0ad10ba2e93bf052482f33325.pdf",
        "target": "409.7",
    },
    {
        "label": "TransUnion intangible assets FY2021 (known-failing control)",
        "question": 'What was the intangible assets of "TransUnion" in the fiscal year 2021?',
        "source": "85fb23ba2910de45e27f8f40170c0f3576043916.pdf",
        "target": "3,770.6",
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
