"""Answerable-questions eval: prototype index vs production index, head-to-head.

Runs the REAL production query path (entity resolution → hybrid retrieval →
70b LLM rerank → parent-page expansion → 8b generation) twice per question —
once with production deps (full 20-filing index) and once with the markdown-
prototype index (5 filings) swapped into QueryDeps — and scores both with the
real eval matcher (scale-tolerant numerics, refusal rules).

Question set: the 3 benchmark items whose companies all live in the slice and
whose golden answers are real (not N/A) — #11 compare, #17 Petra, #25
TransUnion — plus 3 synthetic items from the smoke test's hand-verified
targets. 6 questions x 2 pipelines ~= 120k LLM tokens; the eval runner's
429 self-pacing applies, so expect minutes, not seconds.

Usage (from backend/):  uv run python dev/eval_prototype_slice.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from prototype_markdown_chunking import DATA_DIR, chunk_page, parse_pages  # noqa: E402

from core.config import get_settings  # noqa: E402
from eval.golden import GoldenItem  # noqa: E402
from eval.runner import run_mode  # noqa: E402
from rag.providers.factory import get_embeddings, get_llm  # noqa: E402
from rag.query import QueryDeps, build_deps  # noqa: E402
from rag.retrieval.entity_resolver import build_index  # noqa: E402
from rag.retrieval.reranker import build_reranker  # noqa: E402
from rag.retrieval.store import ChromaStore  # noqa: E402
from smoke_test_retrieval import CASES  # noqa: E402

BENCHMARK_PICKS = [11, 17, 25]  # slice-company questions with real answers

# Hand-verified targets from the retrieval smoke test, as golden items.
# Scales follow each filing's stated units (JPY millions / CHF thousands / USD millions).
SYNTHETIC = [
    GoldenItem(question="What was the shareholders' equity of Mitsui O.S.K. Lines in fiscal year 2022?",
               answer_type="number", answers=[1274570000000], category="synthetic"),
    GoldenItem(question="What was the total shareholders' equity of Compagnie Financiere Tradition SA in fiscal year 2022?",
               answer_type="number", answers=[146469000], category="synthetic"),
    GoldenItem(question="What was the free cash flow of Sensata in fiscal year 2021?",
               answer_type="number", answers=[409700000], category="synthetic"),
]


def load_items() -> list[GoldenItem]:
    rows = json.loads((Path("data/benchmark/answers.json")).read_text(encoding="utf-8"))
    picked = [GoldenItem(question=rows[i]["question"],
                         answer_type=rows[i].get("schema", "number"),
                         answers=rows[i].get("answer", []),
                         category=rows[i].get("category", "other"))
              for i in BENCHMARK_PICKS]
    return picked + SYNTHETIC


def build_prototype_deps(settings, scratch: Path) -> QueryDeps:
    all_docs = []
    for source in sorted({c["source"] for c in CASES}):
        for p in parse_pages(DATA_DIR / source):
            all_docs.extend(chunk_page(p, source, settings.chunk_size))
    store = ChromaStore(get_embeddings(settings), str(scratch), "proto_eval")
    for i in range(0, len(all_docs), 5000):
        store.add(all_docs[i:i + 5000])
    meta = json.loads(Path(settings.doc_metadata_path).read_text(encoding="utf-8"))
    llm = get_llm(settings)
    return QueryDeps(store=store, docstore_docs=all_docs,
                     entity_index=build_index(meta), llm=llm, settings=settings,
                     reranker=build_reranker(settings, get_llm(settings, settings.reranker_llm_model)))


def main() -> None:
    settings = get_settings()
    items = load_items()
    scratch = Path(tempfile.mkdtemp(prefix="proto_eval_"))
    try:
        print("Building prototype index...")
        proto_deps = build_prototype_deps(settings, scratch)
        print("Loading production deps...")
        prod_deps = build_deps(settings)

        print(f"\nRunning {len(items)} answerable questions through BOTH pipelines (hybrid mode)...")
        proto_rows = run_mode(items, "hybrid", proto_deps)
        prod_rows = run_mode(items, "hybrid", prod_deps)

        print("\n" + "=" * 100)
        for item, pr, gd in zip(items, proto_rows, prod_rows):
            print(f"\n[{item.category}] {item.question[:88]}")
            print(f"  golden: {item.answers}")
            for tag, row in (("prototype ", pr), ("production", gd)):
                status = "CORRECT" if row.correct else ("ERROR" if row.error else "WRONG")
                detail = row.error[:70] if row.error else f"refused={row.refused} predicted={row.predicted[:60]!r}"
                print(f"  {tag}: {status:8s} {detail}")
        p_ok = sum(r.correct for r in proto_rows)
        g_ok = sum(r.correct for r in prod_rows)
        print("\n" + "=" * 100)
        print(f"TOTALS — prototype: {p_ok}/{len(items)}   production: {g_ok}/{len(items)}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
