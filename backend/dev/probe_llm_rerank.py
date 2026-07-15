"""Probe: on the SAME fused candidate pool the cross-encoder reranks today,
would an LLM reranker (reference-project technique) surface the table-row
chunks that bge-reranker-base buries?

Reuses the prototype's row-chunk loader + throwaway index (embeddings are
disk-cached, so the rebuild is fast). For each vertical-slice case, scores
every fused candidate two ways -- cross-encoder vs one batched LLM call
returning a 0-1 relevance score per excerpt -- and reports the target's rank
under each.

Snippet length matters: at 300 chars the MITSUI row's FY2022 value (char 329
of a decade-wide row) was truncated away and the LLM correctly deprioritized
it; at 500 chars it ranks #1. Wide multi-year rows bury the current year at
the row's end.

Usage (from backend/):  uv run python dev/probe_llm_rerank.py
Needs GROQ_API_KEY (same as production generation).
"""
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import prototype_table_row_chunking as proto  # noqa: E402

import rag.retrieval.hybrid as hybrid  # noqa: E402
from core.config import get_settings  # noqa: E402
from rag.providers.factory import get_embeddings, get_llm  # noqa: E402
from rag.retrieval.reranker import load_reranker  # noqa: E402
from rag.retrieval.store import ChromaStore  # noqa: E402

_PROMPT = """You are ranking retrieved excerpts from a company's annual report \
by how useful each is for answering the question. An excerpt is highly relevant \
(score near 1.0) only if it contains the specific figure or fact the question \
asks for, for the right period. Excerpts that merely mention related words \
score low. Score EVERY excerpt id exactly once.

Question: {question}

Excerpts:
{excerpts}

Respond with ONLY a JSON array, no prose, one entry per excerpt:
[{{"id": 0, "score": 0.0}}, ...]
"""


_BATCH = 30  # Groq free tier: 6000 tokens/min -- keep each call ~3.5k tokens
_POOL_CAP = 60  # score only the top fused candidates (targets all sit <= 43)
_SNIPPET = 400  # MITSUI's answer value sits at char 329 of its row -- keep > that
_last_call = [0.0]


def _score_batch(llm, question: str, docs: list, offset: int) -> dict[int, float]:
    lines = []
    for i, d in enumerate(docs):
        snippet = " ".join(d.page_content.split())[:_SNIPPET]
        lines.append(f"{offset + i}: {snippet}")
    wait = 35 - (time.monotonic() - _last_call[0])  # pace under the TPM limit
    if wait > 0:
        time.sleep(wait)
    raw = llm.invoke(
        _PROMPT.format(question=question, excerpts="\n".join(lines))).content
    _last_call[0] = time.monotonic()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    entries = json.loads(match.group(0)) if match else []
    return {int(e["id"]): float(e["score"]) for e in entries}


def llm_scores(llm, question: str, docs: list) -> list[float]:
    by_id: dict[int, float] = {}
    pool = docs[:_POOL_CAP]
    for start in range(0, len(pool), _BATCH):
        by_id.update(_score_batch(llm, question, pool[start:start + _BATCH], start))
    return [by_id.get(i, 0.0) for i in range(len(docs))]


def report(tag: str, docs: list, scores, target: str, top_n: int) -> None:
    ranked = sorted(zip(docs, scores), key=lambda p: p[1], reverse=True)
    target_rank = next((i + 1 for i, (d, _) in enumerate(ranked)
                        if target in d.page_content), None)
    verdict = (f"target rank {target_rank}/{len(ranked)}"
               f" -> {'SURVIVES' if target_rank and target_rank <= top_n else 'CUT'}"
               if target_rank else "target NOT IN POOL")
    print(f"  {tag:14s} {verdict}")
    for i, (d, s) in enumerate(ranked[:top_n]):
        mark = " <== TARGET" if target in d.page_content else ""
        ctype = d.metadata.get("chunk_type", "?")
        print(f"      #{i + 1}  {s:+.3f} [{ctype:5s}] "
              f"{' '.join(d.page_content.split())[:80]}{mark}")


def main() -> None:
    settings = get_settings()
    llm = get_llm(settings)
    sources = {c["source"] for c in proto.CASES}

    scratch = Path(tempfile.mkdtemp(prefix="llm_rerank_probe_"))
    try:
        store = ChromaStore(get_embeddings(settings), str(scratch), "probe")
        all_docs = []
        for source in sources:
            pages, rows = proto.load_with_row_tables(proto.DATA_DIR / source)
            chunks = proto.chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
            for c in chunks:
                c.metadata["chunk_type"] = "prose"
            all_docs.extend(chunks)
            all_docs.extend(rows)
        for i in range(0, len(all_docs), 5000):  # Chroma max batch is 5461
            store.add(all_docs[i:i + 5000])
        reranker = load_reranker(settings.reranker_model)

        for case in proto.CASES:
            source, target = case["source"], case["target_substring"]
            print("\n" + "=" * 72)
            print(case["label"])
            fused = hybrid.build_hybrid_retriever(
                store, all_docs, [source], settings.top_k,
                settings.bm25_weight, settings.vector_weight).invoke(case["question"])

            ce = reranker.predict([(case["question"], d.page_content) for d in fused])
            report("cross-encoder:", fused, ce, target, settings.top_n)

            lscores = llm_scores(llm, case["question"], fused)
            report("LLM rerank:", fused, lscores, target, settings.top_n)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
