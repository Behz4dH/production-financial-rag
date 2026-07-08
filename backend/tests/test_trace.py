"""TraceRecorder: records ordered steps with elapsed time."""

from rag.trace import TraceRecorder


def test_record_appends_steps_in_order():
    t = TraceRecorder()
    t.record("parse_query", companies=["Petra"], fiscal_year="2022")
    t.record("resolve_entities", sources=["petra.pdf"], unresolved=[])

    assert len(t.steps) == 2
    assert t.steps[0].stage == "parse_query"
    assert t.steps[0].data == {"companies": ["Petra"], "fiscal_year": "2022"}
    assert t.steps[1].stage == "resolve_entities"
    assert t.steps[0].elapsed_ms >= 0
    assert t.steps[1].elapsed_ms >= t.steps[0].elapsed_ms
