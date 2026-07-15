"""Render a detected table into atomic, self-contained row lines.

10-K financial tables are dense grids: a header band naming the columns
(periods, "Gross"/"Net", …) over data rows whose first cell is the line-item
label. `.to_markdown()` + the recursive character splitter tears these apart,
leaving numbers stranded from their labels. Instead, each *data* row becomes
one atomic chunk carrying its own row label and column headers:

    "Total shareholders' equity -- 31.12.2022: 146,469; 31.12.2021: 187,780"

One row -> one chunk, so a table can never be sliced mid-row. No LLM; the
header/data split is a numeric-density heuristic. (Prefixing rows with the
table's caption was tried and rejected — it worsened both vector and
cross-encoder ranking without helping the LLM reranker.)
"""

import re

# A "clean" financial number: optional currency/sign/parens, comma-grouped or
# decimal, optional percent. Used only to tell a data row from a header row.
_CLEAN_NUMBER = re.compile(r"\(?[¥$€£]?-?\d{1,3}(,\d{3})*(\.\d+)?%?\)?")


def _is_clean_number(cell: str) -> bool:
    cell = cell.strip()
    return bool(cell) and bool(_CLEAN_NUMBER.fullmatch(cell))


def _looks_like_data_row(row: list) -> bool:
    """A data row is mostly numbers; a header row is mostly words."""
    cells = [str(c).strip() for c in row if c and str(c).strip()]
    if not cells:
        return False
    numeric = sum(1 for c in cells if _is_clean_number(c))
    return numeric / len(cells) >= 0.4


def _is_financial_number(cell: str) -> bool:
    """A number that looks like a financial figure — comma-grouped, decimal,
    parenthesized, currency-prefixed, or a percentage. Bare integers ("146",
    "2022") deliberately don't count: page numbers and years are the numbers
    that page-header fragments contain."""
    cell = cell.strip()
    return _is_clean_number(cell) and any(ch in cell for ch in ",.%()$€£¥")


def has_data_rows(rows: list[list]) -> bool:
    """True if the grid holds actual financial data: at least one numeric-dense
    row AND at least one financial-looking figure anywhere. Rejects both prose
    that a text-strategy detector over-boxed and page-header fragments whose
    only numbers are a page number and a year — those junk rows would flood
    BM25 with exactly the tokens every query contains (company name + year)."""
    if not any(_looks_like_data_row(r) for r in rows):
        return False
    return any(_is_financial_number(str(c)) for r in rows for c in r if c)


def render_table_rows(rows: list[list]) -> list[str]:
    """Split header rows (top) from data rows, render each data row as one
    self-contained 'label -- header: value; header: value' line.

    `rows` is the raw cell grid from PyMuPDF `table.extract()`.
    """
    header_end = 0
    for i, row in enumerate(rows[:4]):  # cap: don't hunt for headers forever
        if _looks_like_data_row(row):
            header_end = i
            break
    else:
        header_end = min(1, len(rows))  # no data-like row in first 4 -> 1 header row

    header_rows = rows[:header_end] or ([rows[0]] if rows else [])
    data_rows = rows[header_end:] if header_end else rows[1:]

    n_cols = max((len(r) for r in rows), default=0)
    col_headers = []
    for c in range(n_cols):
        parts = [str(r[c]).strip() for r in header_rows
                 if c < len(r) and r[c] and str(r[c]).strip()]
        col_headers.append(" ".join(parts))

    out: list[str] = []
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
