"""Entity resolution: name/year matching (pure) + LLM query parse (mocked)."""

from rag.retrieval import entity_resolver as R
from rag.retrieval.entity_resolver import QueryEntities


_META = {
    "cross.pdf": {"company_name": "CrossFirst Bankshares, Inc.", "aliases": ["CrossFirst Bank"],
                  "fiscal_year": "2022"},
    "tru.pdf": {"company_name": "TransUnion", "aliases": [], "fiscal_year": "2022"},
}


def _index():
    return R.build_index(_META)


def test_index_includes_aliases():
    names = {r["name"] for r in _index()}
    assert "CrossFirst Bankshares, Inc." in names
    assert "CrossFirst Bank" in names  # alias indexed


def test_resolve_matches_alias_and_correct_year():
    e = QueryEntities(companies=["CrossFirst Bank"], fiscal_year="2022")
    res = R.resolve(e, _index())
    assert res.sources == ["cross.pdf"]
    assert res.unresolved == []
    assert res.refuse is False


def test_resolve_wrong_year_is_unresolved_refusal():
    e = QueryEntities(companies=["CrossFirst Bank"], fiscal_year="2023")
    res = R.resolve(e, _index())
    assert res.sources == []
    assert res.unresolved == ["CrossFirst Bank"]
    assert res.refuse is True


def test_resolve_unknown_company_refuses():
    e = QueryEntities(companies=["Nonexistent Corp"], fiscal_year="2022")
    res = R.resolve(e, _index())
    assert res.refuse is True
    assert res.unresolved == ["Nonexistent Corp"]


def test_resolve_no_year_asked_matches_on_name():
    e = QueryEntities(companies=["TransUnion"], fiscal_year=None)
    res = R.resolve(e, _index())
    assert res.sources == ["tru.pdf"]


def test_parse_query_uses_structured_output():
    class _Structured:
        def invoke(self, _p):
            return QueryEntities(companies=["TransUnion"], fiscal_year="2022")

    class _LLM:
        def with_structured_output(self, _schema):
            return _Structured()

    parsed = R.parse_query('total assets of "TransUnion" in 2022?', _LLM())
    assert parsed.companies == ["TransUnion"]
    assert parsed.fiscal_year == "2022"
