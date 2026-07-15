"""Render markdown pipe-tables into atomic, caption'd row lines.

Financial tables arrive from pymupdf4llm as markdown: a header band (possibly
bold, possibly <br>-split across lines) over data rows whose first wordy cell
is the line-item label. Each DATA row becomes one self-contained line:

    "Consolidated Income Statement — Profit for the Year: 2022: 88.1; Restated 2021: 196.6"

One row -> one chunk: term-dense for BM25 and the embedder, caption and
column headers for context (the units note and period labels were exactly
what the geometric renderer lost, causing correct-but-needless generation
refusals). Markup is stripped for the indexed text: BM25 tokenizes on
whitespace, so `equity**|` never matches `equity`, and pipes/bold pollute
embeddings.
"""

import re

# A cell that is purely numeric/currency/punctuation — NOT a line-item label.
_NUMERIC_CELL = re.compile(r"[\d.,()%$€£¥\s\-]+")
_SEPARATOR_LINE = re.compile(r"\s*\|[\s:|\-]+\|?\s*$")


def plain_text(text: str) -> str:
    """Strip markdown markup (bold, pipes, headings, <br>, separator dashes),
    keep the content, collapse whitespace."""
    text = text.replace("**", "").replace("<br>", " ")
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\|[\s:|\-]+\|", " ", text)  # separator rows
    return " ".join(text.replace("|", " ").split())


def render_markdown_table_rows(table_lines: list[str], caption: str) -> list[str]:
    """One 'caption — label: header: value; ...' line per data row.

    ``table_lines`` is a run of markdown table lines (each starting with
    ``|``); ``caption`` is the nearest preceding heading/sentence, already
    markup-stripped by the caller (may be empty).
    """
    lines = [ln for ln in table_lines if ln.lstrip().startswith("|")]
    if not lines:
        return []
    sep = next((i for i, ln in enumerate(lines) if _SEPARATOR_LINE.match(ln)), 1)
    headers = [plain_text(cell) for cell in "|".join(lines[:sep]).split("|")]

    out: list[str] = []
    for ln in lines[sep + 1:]:
        cells = [plain_text(cell) for cell in ln.strip().strip("|").split("|")]
        if not any(cells):
            continue
        label = next((c for c in cells[:2] if c and not _NUMERIC_CELL.fullmatch(c)), "")
        pairs = []
        for i, value in enumerate(cells):
            if not value or value == label:
                continue
            header = headers[i + 1] if i + 1 < len(headers) else ""
            pairs.append(f"{header}: {value}" if header else value)
        if not pairs:
            continue
        head = f"{caption} — {label}" if caption and label else (caption or label)
        out.append((f"{head}: " if head else "") + "; ".join(pairs))
    return out
