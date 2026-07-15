"""Prototype: markdown-first parsing + structure chunking (the proposed v2).

Design under test, on the 5-case vertical slice from smoke_test_retrieval:
  1. pymupdf4llm renders each page to markdown (tables come out as pipe
     tables with labels/units intact -- replaces find_tables + subtraction +
     the genuine/data-row heuristics).
  2. Running headers/footers (lines on >=40% of a doc's pages) are stripped.
  3. Structure chunking: markdown tables are ATOMIC chunks, prefixed with
     their nearest preceding heading/caption; oversized tables split into
     row-groups that repeat the header. Prose packs paragraphs to ~chunk_size.
  4. Parent-page check: would generation see the answer if each surviving
     chunk were expanded to its full page (the retrieve-small/generate-big
     move)?

Measures the same stages as dev/smoke_test_retrieval.py (BM25 / vector /
fused / LLM rerank) so the two runs are directly comparable.

Usage (from backend/):  uv run python dev/prototype_markdown_chunking.py
Needs GROQ_API_KEY (LLM rerank arm) and `uv pip install pymupdf4llm`.
"""

import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pymupdf4llm  # noqa: E402
from langchain_community.retrievers import BM25Retriever  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

import rag.retrieval.hybrid as hybrid  # noqa: E402
from core.config import get_settings  # noqa: E402
from rag.providers.factory import get_embeddings, get_llm  # noqa: E402
from rag.retrieval.reranker import build_reranker, rerank  # noqa: E402
from rag.retrieval.store import ChromaStore  # noqa: E402

from smoke_test_retrieval import CASES  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"
MAX_TABLE_CHARS = 2500   # bigger tables split into row-groups w/ repeated header
ROW_GROUP = 15


# --- 1+2: parse to per-page markdown, strip running headers/footers ---------

def parse_pages(path: Path) -> list[dict]:
    """[{page, text}] with document-boilerplate lines removed."""
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
    norm = lambda ln: " ".join(ln.split())  # noqa: E731
    freq = Counter()
    for p in pages:
        for ln in {norm(ln) for ln in p["text"].splitlines() if ln.strip()}:
            freq[ln] += 1
    n_pages = len(pages)
    boiler = {ln for ln, c in freq.items()
              if c >= max(5, 0.4 * n_pages) and not ln.startswith("|")}
    out = []
    for p in pages:
        kept = [ln for ln in p["text"].splitlines()
                if norm(ln) not in boiler
                and not re.fullmatch(r"[\d\s\-–—]*", norm(ln))]  # bare page numbers
        out.append({"page": p["metadata"]["page_number"], "text": "\n".join(kept)})
    return out


# --- 3: structure chunking ---------------------------------------------------

def _split_blocks(text: str) -> list[tuple[str, str]]:
    """[(kind, block)] where kind is 'table' or 'prose'; tables are runs of
    '|' lines, each glued to its nearest preceding heading/caption line."""
    lines = text.splitlines()
    blocks: list[tuple[str, str]] = []
    prose: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            table = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                table.append(lines[i])
                i += 1
            caption = ""
            for prev in reversed(prose):
                if prev.strip():
                    caption = prev.strip()
                    break
            if prose:
                blocks.append(("prose", "\n".join(prose)))
                prose = []
            blocks.append(("table", (caption + "\n" if caption else "") + "\n".join(table)))
        else:
            prose.append(lines[i])
            i += 1
    if prose:
        blocks.append(("prose", "\n".join(prose)))
    return blocks


def _split_big_table(block: str) -> list[str]:
    lines = block.splitlines()
    head_end = next((k for k, ln in enumerate(lines)
                     if ln.lstrip().startswith("|---")), 1) + 1
    header, body = lines[:head_end], lines[head_end:]
    parts = []
    for start in range(0, len(body), ROW_GROUP):
        parts.append("\n".join(header + body[start:start + ROW_GROUP]))
    return parts or [block]


def _plain(text: str) -> str:
    """Strip markdown markup for the INDEXED text (BM25 tokenizes on
    whitespace — `equity**|` never matches `equity`; pipes/bold also pollute
    embeddings). The pretty markdown stays available via parent-page lookup."""
    text = text.replace("**", "").replace("<br>", " ")
    return " ".join(text.replace("|", " ").split())


def _table_row_docs(block: str, caption_fallback: str) -> list[str]:
    """One term-dense index unit per data row: 'caption — label: hdr val; ...'.

    Indexing whole tables measured badly here: a 1,100-char balance sheet
    mentions the query's words once or twice among hundreds of number tokens,
    so BM25's frequency/length math buries it under prose. Small row units are
    the lexical fix; the caption carries the semantic context."""
    lines = [ln for ln in block.splitlines() if ln.lstrip().startswith("|")]
    caption = next((ln for ln in block.splitlines()
                    if not ln.lstrip().startswith("|") and ln.strip()), caption_fallback)
    caption = _plain(re.sub(r"^#+\s*", "", caption))
    sep = next((i for i, ln in enumerate(lines) if re.match(r"\s*\|[\s:|-]+$", ln)), 1)
    headers = [_plain(c) for c in "|".join(lines[:sep]).split("|")]
    out = []
    for ln in lines[sep + 1:]:
        cells = [_plain(c) for c in ln.strip().strip("|").split("|")]
        label = next((c for c in cells[:2] if c and not re.fullmatch(r"[\d.,()%$€£¥\s-]+", c)), "")
        pairs = []
        for i, val in enumerate(cells):
            if not val or val == label:
                continue
            hdr = headers[i + 1] if i + 1 < len(headers) else ""
            pairs.append(f"{hdr}: {val}" if hdr else val)
        if pairs:
            head = f"{caption} — {label}" if label else caption
            out.append(f"{head}: " + "; ".join(pairs))
    return out


def chunk_page(page: dict, source: str, chunk_size: int) -> list[Document]:
    docs: list[Document] = []
    counters = {"table": 0, "prose": 0}
    last_heading = ""

    def emit(kind: str, text: str) -> None:
        if not text.strip():
            return
        idx = counters[kind]
        counters[kind] += 1
        docs.append(Document(page_content=text.strip(), metadata={
            "source": source, "page": page["page"], "chunk_type": kind,
            "chunk_id": f"{source}::p{page['page']}::{'t' if kind == 'table' else 'c'}{idx}"}))

    for kind, block in _split_blocks(page["text"]):
        if kind == "table":
            for row in _table_row_docs(block, last_heading):
                emit("table", row)
        else:
            m = re.findall(r"^#+\s*(.+)$", block, re.M)
            if m:
                last_heading = m[-1]
            # pack paragraphs (split on blank lines / headings) up to chunk_size
            buf = ""
            for para in re.split(r"\n\s*\n|(?=\n#)", block):
                if len(buf) + len(para) > chunk_size and buf:
                    emit("prose", _plain(buf))
                    buf = ""
                buf += para + "\n"
            emit("prose", _plain(buf))
    return docs


# --- 4: measure ---------------------------------------------------------------

def _rank_of(target: str, docs: list) -> str:
    for i, d in enumerate(docs):
        if target in d.page_content:
            return f"FOUND at #{i + 1} of {len(docs)}  [{d.metadata.get('chunk_type')}]"
    return f"MISSING (of {len(docs)})"


def main() -> None:
    settings = get_settings()
    reranker = build_reranker(settings, get_llm(settings, settings.reranker_llm_model))
    sources = {c["source"] for c in CASES}

    scratch = Path(tempfile.mkdtemp(prefix="md_proto_"))
    try:
        print("Parsing + chunking slice filings (pymupdf4llm)...")
        all_docs: list[Document] = []
        page_text: dict[tuple[str, int], str] = {}  # parent-page lookup
        for source in sorted(sources):
            pages = parse_pages(DATA_DIR / source)
            n_before = len(all_docs)
            for p in pages:
                page_text[(source, p["page"])] = p["text"]
                all_docs.extend(chunk_page(p, source, settings.chunk_size))
            print(f"  {source[:12]}...  {len(pages)} pages -> {len(all_docs) - n_before} chunks")

        kinds = Counter(d.metadata["chunk_type"] for d in all_docs)
        print(f"Total: {len(all_docs)} chunks ({dict(kinds)})")

        store = ChromaStore(get_embeddings(settings), str(scratch), "md_proto")
        for i in range(0, len(all_docs), 5000):
            store.add(all_docs[i:i + 5000])

        for case in CASES:
            source, target = case["source"], case["target"]
            subset = [d for d in all_docs if d.metadata["source"] == source]
            print("\n" + "=" * 72)
            print(case["label"])

            bm25 = BM25Retriever.from_documents(subset)
            bm25.k = settings.top_k
            print("  BM25 alone:   ", _rank_of(target, bm25.invoke(case["question"])))

            vector = hybrid.vector_only_retriever(store, [source], settings.top_k)
            print("  Vector alone: ", _rank_of(target, vector.invoke(case["question"])))

            fused = hybrid.build_hybrid_retriever(
                store, all_docs, [source], settings.top_k,
                settings.bm25_weight, settings.vector_weight).invoke(case["question"])
            print("  Fused hybrid: ", _rank_of(target, fused))

            kept = rerank(case["question"], fused, reranker, settings.top_n)
            print("  LLM rerank:   ", _rank_of(target, kept),
                  " top_score:", round(kept[0].metadata.get("rerank_score", 0), 2) if kept else "-")

            # parent-page expansion: would generation see the answer?
            expanded = any(target in page_text.get((source, d.metadata["page"]), "")
                           for d in kept)
            print("  Parent-page:  ", "ANSWER VISIBLE to generation" if expanded
                  else "answer still not visible")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
