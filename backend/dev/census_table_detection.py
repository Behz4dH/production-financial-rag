"""Corpus census: per filing, how many genuine tables does the production
default find_tables() detect vs strategy='text'? Filings where the default
finds ~none are silently degraded (tables sliced as raw narrative), like the
TransUnion 10-K turned out to be.

'text'-strategy counts are inflated (it over-boxes prose), so read them as an
upper bound; the signal is a default count in the low double digits on a
100+ page financial report.

Usage (from backend/):  uv run python dev/census_table_detection.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pymupdf  # noqa: E402

from rag.ingestion.loaders import _is_genuine_table  # noqa: E402

DATA_DIR = Path("data/docs")
META_PATH = Path("data/doc_metadata.json")


def company_of(meta: dict, fname: str) -> str:
    entry = meta.get(fname, {})
    if isinstance(entry, dict):
        return str(entry.get("company", entry.get("company_name", "?")))
    return "?"


def census(doc, strategy=None):
    n_tables = n_rows = 0
    for page in doc:
        tf = page.find_tables(strategy=strategy) if strategy else page.find_tables()
        for t in tf.tables:
            if _is_genuine_table(t):
                n_tables += 1
                n_rows += t.row_count
    return n_tables, n_rows


def main() -> None:
    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    print(f"{'filing':44s} {'pages':>5s} {'default':>14s} {'text':>14s}  company")
    for pdf in sorted(DATA_DIR.glob("*.pdf")):
        doc = pymupdf.open(str(pdf))
        try:
            dt, dr = census(doc)
            tt, tr = census(doc, "text")
            flag = "  <== DEGRADED" if dt <= max(2, tt // 20) else ""
            print(f"{pdf.name:44s} {doc.page_count:5d} {dt:5d}t/{dr:6d}r {tt:5d}t/{tr:6d}r  "
                  f"{company_of(meta, pdf.name)[:30]}{flag}")
        finally:
            doc.close()


if __name__ == "__main__":
    main()
