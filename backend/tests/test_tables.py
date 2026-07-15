"""Markdown pipe-tables -> atomic, caption'd, self-contained row lines.

pymupdf4llm renders financial tables as markdown with labels/units/headings
intact; each DATA row becomes one term-dense index unit carrying the table's
caption and its column headers. Markup is stripped for indexing (BM25
tokenizes on whitespace — `equity**|` never matches `equity`)."""

from rag.ingestion.tables import plain_text, render_markdown_table_rows

INCOME_STATEMENT = """|US$ million|Notes|**2022**|Restated 2021 ¹|
|---|---|---|---|
|Revenue|2|**585.2**|406.9|
|**Profit for the Year**||**88.1**|196.6|""".splitlines()


def test_each_data_row_becomes_one_captioned_line():
    rows = render_markdown_table_rows(INCOME_STATEMENT, "Consolidated Income Statement")
    assert rows == [
        "Consolidated Income Statement — Revenue: Notes: 2; 2022: 585.2; Restated 2021 ¹: 406.9",
        "Consolidated Income Statement — Profit for the Year: 2022: 88.1; Restated 2021 ¹: 196.6",
    ]


def test_rows_without_caption_still_render():
    rows = render_markdown_table_rows(INCOME_STATEMENT, "")
    assert rows[0].startswith("Revenue:")


def test_multiline_headers_and_bold_markup_are_flattened():
    table = """||**December 31,**<br>**2022**|**December 31,**<br>**2021**|
|---|---|---|
|Total intangible assets|$5,944.1|$5,679.5|""".splitlines()
    rows = render_markdown_table_rows(table, "Note 7")
    assert rows == ["Note 7 — Total intangible assets: "
                    "December 31, 2022: $5,944.1; December 31, 2021: $5,679.5"]


def test_numeric_label_cells_are_not_mistaken_for_labels():
    table = """|Metric|2022|
|---|---|
|409.7|1,000.0|""".splitlines()
    rows = render_markdown_table_rows(table, "")
    # no wordy label cell -> values render without a fake label
    assert rows == ["Metric: 409.7; 2022: 1,000.0"]


def test_empty_and_separator_rows_are_skipped():
    table = """|A|B|
|---|---|
||||
|Assets|5,679.5|""".splitlines()
    rows = render_markdown_table_rows(table, "")
    assert rows == ["Assets: B: 5,679.5"]


def test_plain_text_strips_markup_but_keeps_content():
    md = "###### **Total** shareholders' equity | 146,469 |<br>next"
    out = plain_text(md)
    assert "**" not in out and "|" not in out and "<br>" not in out and "#" not in out
    assert "Total shareholders' equity" in out
    assert "146,469" in out
