"""Prototype: Option B from the retrieval scoping discussion -- no LLM table
serialization, just stop losing the header-to-value association that
`.to_markdown()` currently throws away, and chunk each table row as its own
atomic unit (never sliced by the recursive character splitter).

For each genuine table (same PyMuPDF `find_tables()` detection already used
in `rag/ingestion/loaders.py`): pull raw cells via `.extract()` instead of
`.to_markdown()`, heuristically split header rows from data rows (a row is
"data" if enough of its cells look like clean financial numbers), then render
each data row as one self-contained line: "{row label} -- {col header}:
{value}; {col header}: {value}; ...". One row -> one chunk, so a table can
never get sliced mid-row by the generic splitter.

Builds a throwaway Chroma index (NOT the production data/chroma) for just the
5 filings in our vertical slice, then re-runs the same BM25/vector/fused/
rerank recall check from smoke_test_retrieval.py against it, so the before/
after numbers are directly comparable.

Usage (from backend/):  uv run python dev/prototype_table_row_chunking.py
"""

import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pymupdf  # noqa: E402
from langchain_community.retrievers import BM25Retriever  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

import rag.retrieval.hybrid as hybrid  # noqa: E402
from core.config import get_settings  # noqa: E402
from rag.ingestion.chunking import chunk_pages  # noqa: E402
from rag.ingestion.loaders import _is_genuine_table  # noqa: E402
from rag.providers.factory import get_embeddings  # noqa: E402
from rag.retrieval.reranker import load_reranker, rerank  # noqa: E402
from rag.retrieval.store import ChromaStore  # noqa: E402

DATA_DIR = Path("data/docs")

CASES = [
    {
        "label": "MITSUI shareholders' equity FY2022",
        "question": "What was the shareholders' equity of Mitsui O.S.K. Lines in fiscal year 2022?",
        "source": "6054ec55767fbe6585598ced7afacf5cb8619a13.pdf",
        "target_substring": "1,274,570",
    },
    {
        "label": "Tradition shareholders' equity FY2022",
        "question": "What was the total shareholders' equity of Compagnie Financiere Tradition SA in fiscal year 2022?",
        "source": "2779336b845a41544348abb7b3e6e5bd2ff893a2.pdf",
        "target_substring": "146,469",
    },
    {
        "label": "Petra Diamonds net income FY2022 (regression check)",
        "question": "What was the net income of Petra Diamonds in fiscal year 2022?",
        "source": "609042c64a759c0ac63e7cf18742be4dd3cc5cd5.pdf",
        "target_substring": "88.1",
    },
    {
        "label": "Sensata free cash flow FY2021 (regression check)",
        "question": "What was the free cash flow of Sensata in fiscal year 2021?",
        "source": "e33544bdea57faa0ad10ba2e93bf052482f33325.pdf",
        "target_substring": "409.7",
    },
    {
        "label": "TransUnion intangible assets FY2021 (known-failing control)",
        "question": 'What was the intangible assets of "TransUnion" in the fiscal year 2021?',
        "source": "85fb23ba2910de45e27f8f40170c0f3576043916.pdf",
        "target_substring": "3,770.6",
    },
]

_CLEAN_NUMBER = re.compile(r"\(?[¥$€£]?-?\d{1,3}(,\d{3})*(\.\d+)?%?\)?")


def _is_clean_number(cell: str) -> bool:
    cell = cell.strip()
    return bool(cell) and bool(_CLEAN_NUMBER.fullmatch(cell))


def _looks_like_data_row(row: list) -> bool:
    cells = [str(c).strip() for c in row if c and str(c).strip()]
    if not cells:
        return False
    numeric = sum(1 for c in cells if _is_clean_number(c))
    return numeric / len(cells) >= 0.4


def _table_to_row_texts(rows: list[list]) -> list[str]:
    """Split into header rows (top) + data rows, render each data row as one
    self-contained 'label -- header: value; header: value' line."""
    header_end = 0
    for i, row in enumerate(rows[:4]):  # cap: don't hunt for headers forever
        if _looks_like_data_row(row):
            header_end = i
            break
    else:
        header_end = min(1, len(rows))  # no data-like row found in the first 4 -- assume 1 header row

    header_rows = rows[:header_end] or ([rows[0]] if rows else [])
    data_rows = rows[header_end:] if header_end else rows[1:]

    n_cols = max((len(r) for r in rows), default=0)
    col_headers = []
    for c in range(n_cols):
        parts = [str(r[c]).strip() for r in header_rows if c < len(r) and r[c] and str(r[c]).strip()]
        col_headers.append(" ".join(parts))

    out = []
    for row in data_rows:
        cells = [str(c).strip() if c else "" for c in row]
        if not any(cells):
            continue
        row_label = next((c for c in cells[:2] if c), "")
        parts = []
        for c, val in enumerate(cells):
            if not val or val == row_label:
                continue
            header = col_headers[c] if c < len(col_headers) else ""
            parts.append(f"{header}: {val}" if header else val)
        if not parts:
            continue
        text = f"{row_label} -- " + "; ".join(parts) if row_label else "; ".join(parts)
        out.append(text)
    return out


def _detect_tables(page) -> tuple[list, bool]:
    """Line-based detection first (the production default); if a page yields
    no genuine tables, retry with strategy='text' for borderless layouts like
    the TransUnion 10-K (where line-based detection finds ~nothing in the
    whole filing). Text-strategy over-boxes prose as 'tables', and kept tables
    get subtracted from the page's narrative text, so a fallback table is kept
    only if it has >= 2 numeric-looking data rows. Returns (tables, fallback):
    fallback tables need line-based rendering because their extract() cell
    grid is unreliable (words split across cells)."""
    tables = [t for t in page.find_tables().tables if _is_genuine_table(t)]
    if tables:
        return tables, False
    fallback = []
    for t in page.find_tables(strategy="text").tables:
        if not _is_genuine_table(t):
            continue
        if sum(1 for r in t.extract() if _looks_like_data_row(r)) >= 2:
            fallback.append(t)
    return fallback, True


# A financial value (not a bare small int, so date lines don't count):
# currency-prefixed, comma-grouped, or decimal.
_VALUE = re.compile(r"[¥$€£]\s?\(?\d|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+")
_DOT_LEADERS = re.compile(r"(?:\.\s+){2,}\.?")


def _fallback_row_texts(page, bbox) -> list[str]:
    """Row texts for a text-strategy fallback table: raw text lines inside the
    table region (reading order), which keep each row's label and values
    together, prefixed with a heading built from the region's leading
    non-data lines (table title + column-header lines)."""
    raw = page.get_text(clip=bbox, sort=True)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    is_data = [len(_VALUE.findall(ln)) >= 2 for ln in lines]
    first_data = next((i for i, d in enumerate(is_data) if d), None)
    if first_data is None:
        return []
    title = lines[0] if not is_data[0] else ""
    col_headers = " ".join(lines[max(1, first_data - 3):first_data])
    heading = " ".join(x for x in (title, col_headers) if x)[:250]
    out = []
    for ln, d in zip(lines, is_data):
        if not d:
            continue
        ln = _DOT_LEADERS.sub(" ", ln)
        out.append(f"{heading}\n{ln}" if heading else ln)
    return out


def _table_caption(page, bbox, max_chars: int = 200) -> str:
    """Nearest text above the table on the same page -- typically its heading
    and units note (e.g. 'Consolidated Balance Sheets' + '(Millions of yen)').
    This is document-local context, NOT doc_metadata.json attributes, so it
    doesn't denormalize entity metadata onto chunks."""
    above = []
    for b in page.get_text("blocks"):
        r = pymupdf.Rect(b[:4])
        if r.y1 <= bbox.y0 + 2 and b[4].strip():
            above.append((r.y1, b[4].strip().replace("\n", " ")))
    if not above:
        return ""
    above.sort(key=lambda t: t[0])
    text = " ".join(t[1] for t in above[-2:])  # the two blocks closest above
    return text[-max_chars:]


def load_with_row_tables(path: Path) -> tuple[list[Document], list[Document]]:
    """Returns (narrative_pages, table_row_docs) -- narrative still has table
    regions subtracted, same as the production loader; table rows are new."""
    doc = pymupdf.open(str(path))
    narrative_pages: list[Document] = []
    table_rows: list[Document] = []
    try:
        for page in doc:
            tables, is_fallback = _detect_tables(page)
            boxes = [pymupdf.Rect(t.bbox) for t in tables]

            if boxes:
                blocks = page.get_text("blocks")
                narrative = "\n".join(
                    b[4] for b in blocks if not any(pymupdf.Rect(b[:4]) in bx for bx in boxes)
                )
            else:
                narrative = page.get_text()

            narrative_pages.append(Document(page_content=narrative,
                                            metadata={"source": path.name, "page": page.number + 1}))

            for t_idx, table in enumerate(tables):
                if is_fallback:
                    # fallback rows carry their own heading; extract()'s cell
                    # grid is unreliable for these layouts
                    row_texts = _fallback_row_texts(page, pymupdf.Rect(table.bbox))
                else:
                    caption = _table_caption(page, pymupdf.Rect(table.bbox))
                    row_texts = [f"{caption}\n{t}" if caption else t
                                 for t in _table_to_row_texts(table.extract())]
                for r_idx, text in enumerate(row_texts):
                    table_rows.append(Document(
                        page_content=text,
                        metadata={"source": path.name, "page": page.number + 1,
                                 "chunk_id": f"{path.name}::p{page.number + 1}::t{t_idx}::r{r_idx}",
                                 "chunk_type": "table"},
                    ))
        return narrative_pages, table_rows
    finally:
        doc.close()


def _print_full_rerank_diagnostic(question: str, fused_docs: list, reranker, target: str, top_n: int) -> None:
    """Score every fused candidate (not just the top_n kept), so we can see
    where the target actually sits in the cross-encoder's score distribution
    and what specifically outscored it into the kept top_n."""
    scores = reranker.predict([(question, d.page_content) for d in fused_docs])
    ranked = sorted(zip(fused_docs, scores), key=lambda pair: pair[1], reverse=True)

    print(f"    -- full rerank diagnostic ({len(ranked)} candidates, cutoff is top_n={top_n}) --")
    for i, (doc, score) in enumerate(ranked):
        if i == top_n:
            print("  ---- top_n cutoff ----")
        is_target = target in doc.page_content
        marker = " <== TARGET" if is_target else ""
        if i <= top_n or is_target:
            snippet = doc.page_content[:90].replace("\n", " ")
            ctype = doc.metadata.get("chunk_type", "?")
            print(f"    #{i + 1:2d}  score={score:+.3f}  [{ctype:5s}]  {snippet}{marker}")


def main() -> None:
    settings = get_settings()
    sources = {c["source"] for c in CASES}

    scratch_dir = Path(tempfile.mkdtemp(prefix="table_row_prototype_"))
    print(f"Scratch Chroma dir: {scratch_dir}")
    try:
        store = ChromaStore(get_embeddings(settings), str(scratch_dir), "prototype_table_rows")
        all_docs: list[Document] = []

        for source in sources:
            narrative_pages, table_rows = load_with_row_tables(DATA_DIR / source)
            narrative_chunks = chunk_pages(narrative_pages, settings.chunk_size, settings.chunk_overlap)
            for c in narrative_chunks:
                c.metadata["chunk_type"] = "prose"
            print(f"{source}: {len(narrative_chunks)} narrative chunks + {len(table_rows)} table-row chunks")
            all_docs.extend(narrative_chunks)
            all_docs.extend(table_rows)

        for i in range(0, len(all_docs), 5000):  # Chroma max batch is 5461
            store.add(all_docs[i:i + 5000])
        reranker = load_reranker(settings.reranker_model)

        def _rank_of(substring: str, docs: list) -> str:
            for i, d in enumerate(docs):
                if substring in d.page_content:
                    return f"FOUND at #{i + 1} of {len(docs)}  [{d.metadata.get('chunk_type', '?')}]"
            return f"MISSING (of {len(docs)})"

        for case in CASES:
            source, target = case["source"], case["target_substring"]
            subset = [d for d in all_docs if d.metadata.get("source") == source]

            print("\n" + "=" * 72)
            print(case["label"])
            print(" ", case["question"])

            bm25 = BM25Retriever.from_documents(subset)
            bm25.k = settings.top_k
            print("  BM25 alone:   ", _rank_of(target, bm25.invoke(case["question"])))

            vector_docs = hybrid.vector_only_retriever(store, [source], settings.top_k).invoke(case["question"])
            print("  Vector alone: ", _rank_of(target, vector_docs))

            fused_docs = hybrid.build_hybrid_retriever(
                store, all_docs, [source], settings.top_k,
                settings.bm25_weight, settings.vector_weight).invoke(case["question"])
            print("  Fused hybrid: ", _rank_of(target, fused_docs))

            reranked = rerank(case["question"], fused_docs, reranker, settings.top_n)
            print("  After rerank: ", _rank_of(target, reranked))

            found_in_fused = any(target in d.page_content for d in fused_docs)
            found_after_rerank = any(target in d.page_content for d in reranked)
            if found_in_fused and not found_after_rerank:
                _print_full_rerank_diagnostic(case["question"], fused_docs, reranker, target, settings.top_n)
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
