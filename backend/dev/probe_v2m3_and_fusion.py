"""Two probes on the markdown-prototype slice index:

1. FUSION WEIGHTS (Q: can tuning bm25/vector weights help?)
   With a reranker in the pipeline, fused ORDER is discarded — rerank scores
   the whole pool and keeps its own top_n. So weights can only matter if they
   change pool MEMBERSHIP. EnsembleRetriever fuses the two arms' top-k lists
   by weighted RRF, i.e. the pool is always their UNION — membership should be
   weight-invariant. This probe verifies that empirically across three weight
   splits.

2. bge-reranker-v2-m3 (Q: can a bigger LOCAL cross-encoder replace the 70b?)
   Scores the same fused pools the LLM reranker ranked yesterday and reports
   the target's rank + top-3, for a direct offline-vs-LLM comparison.
   First run downloads ~2.2GB of weights.

Usage (from backend/):  uv run python dev/probe_v2m3_and_fusion.py
"""

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

import rag.retrieval.hybrid as hybrid  # noqa: E402
from core.config import get_settings  # noqa: E402
from rag.providers.factory import get_embeddings  # noqa: E402
from rag.retrieval.store import ChromaStore  # noqa: E402
from smoke_test_retrieval import CASES  # noqa: E402

WEIGHT_SPLITS = [(0.2, 0.8), (0.4, 0.6), (0.8, 0.2)]


def main() -> None:
    settings = get_settings()
    scratch = Path(tempfile.mkdtemp(prefix="v2m3_probe_"))
    try:
        print("Building slice index (embeddings disk-cached from last run)...")
        all_docs = []
        for source in sorted({c["source"] for c in CASES}):
            for p in parse_pages(DATA_DIR / source):
                all_docs.extend(chunk_page(p, source, settings.chunk_size))
        store = ChromaStore(get_embeddings(settings), str(scratch), "v2m3_probe")
        for i in range(0, len(all_docs), 5000):
            store.add(all_docs[i:i + 5000])
        print(f"  {len(all_docs)} index units")

        print("\n--- Probe 1: does fusion weighting change pool membership? ---")
        for case in CASES:
            src = case["source"]
            pools = {}
            for bw, vw in WEIGHT_SPLITS:
                fused = hybrid.build_hybrid_retriever(
                    store, all_docs, [src], settings.top_k, bw, vw).invoke(case["question"])
                pools[(bw, vw)] = frozenset(d.metadata["chunk_id"] for d in fused)
            sets = list(pools.values())
            identical = all(s == sets[0] for s in sets)
            fused_default = hybrid.build_hybrid_retriever(
                store, all_docs, [src], settings.top_k,
                settings.bm25_weight, settings.vector_weight).invoke(case["question"])
            present = any(case["target"] in d.page_content for d in fused_default)
            print(f"  {case['label'][:42]:44s} membership identical across weights: "
                  f"{identical} | target in pool: {present}")

        print("\n--- Probe 2: bge-reranker-v2-m3 (local) on the same pools ---")
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder("BAAI/bge-reranker-v2-m3")
        for case in CASES:
            fused = hybrid.build_hybrid_retriever(
                store, all_docs, [case["source"]], settings.top_k,
                settings.bm25_weight, settings.vector_weight).invoke(case["question"])
            scores = ce.predict([(case["question"], d.page_content) for d in fused])
            ranked = sorted(zip(fused, scores), key=lambda p: p[1], reverse=True)
            rank = next((i + 1 for i, (d, _) in enumerate(ranked)
                         if case["target"] in d.page_content), None)
            verdict = (f"rank {rank}/{len(ranked)} -> "
                       f"{'SURVIVES top_5' if rank and rank <= settings.top_n else 'CUT'}"
                       if rank else "NOT IN POOL")
            print(f"\n  {case['label'][:60]}")
            print(f"    v2-m3: {verdict}")
            for i, (d, s) in enumerate(ranked[:3]):
                mark = "  <== TARGET" if case["target"] in d.page_content else ""
                print(f"      #{i + 1} {s:+.3f} [{d.metadata.get('chunk_type')}] "
                      f"{' '.join(d.page_content.split())[:80]}{mark}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
