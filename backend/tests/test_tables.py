"""render_table_rows: header/data split + one atomic self-contained line per
data row (no captions)."""

from rag.ingestion.tables import render_table_rows


def test_each_data_row_becomes_one_labelled_line():
    grid = [
        ["", "31.12.2022", "31.12.2021"],          # header row (mostly words/dates)
        ["Total shareholders' equity", "146,469", "187,780"],
        ["Revenue", "1,000", "900"],
    ]
    rows = render_table_rows(grid)
    assert rows == [
        "Total shareholders' equity -- 31.12.2022: 146,469; 31.12.2021: 187,780",
        "Revenue -- 31.12.2022: 1,000; 31.12.2021: 900",
    ]


def test_row_label_not_duplicated_as_a_value():
    grid = [
        ["Item", "2022"],
        ["Net income", "88.1"],
    ]
    rows = render_table_rows(grid)
    assert rows == ["Net income -- 2022: 88.1"]


def test_blank_rows_are_skipped():
    grid = [
        ["", "2022"],
        ["", ""],
        ["Assets", "5,679.5"],
    ]
    rows = render_table_rows(grid)
    assert rows == ["Assets -- 2022: 5,679.5"]


def test_no_caption_prefix():
    # Rows carry only their own label + column headers, never a table title.
    grid = [["Metric", "FY2022"], ["Free cash flow", "409.7"]]
    rows = render_table_rows(grid)
    assert rows == ["Free cash flow -- FY2022: 409.7"]


def test_has_data_rows_accepts_financial_grids():
    from rag.ingestion.tables import has_data_rows

    assert has_data_rows([["", "2021", "2020"],
                          ["Intangibles, gross", "5,679.5", "5,516.0"]]) is True
    assert has_data_rows([["Profit for the Year", "88.1", "196.6"]]) is True
    assert has_data_rows([["Costs", "(391.5)", "(356.1)"]]) is True


def test_has_data_rows_rejects_prose_and_header_fragments():
    from rag.ingestion.tables import has_data_rows

    # Pure prose over-boxed by text-strategy detection.
    assert has_data_rows([["Dear shareholders", "this year"],
                          ["we delivered", "strong results"]]) is False
    # Page-header fragment: its only "numbers" are a page number and a year.
    # Bare integers must NOT qualify a grid as financial data — these junk
    # rows otherwise flood BM25 with the query's own tokens (company + year).
    assert has_data_rows([["146", "Petra Diamonds"],
                          ["2022", "Annual Report"]]) is False
