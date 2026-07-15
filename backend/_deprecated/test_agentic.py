"""Agentic path: refusal on unknown entity; answer when resolved (mocked LLM)."""

from langchain_core.documents import Document

from _deprecated.agentic import answer_agentic
from rag.generation.schema import RAGAnswer
from rag.query import QueryDeps
from rag.retrieval.entity_resolver import QueryEntities, build_index
from rag.retrieval.store import ChromaStore
from rag.trace import TraceRecorder
from tests.fakes import HashEmbeddings


class _LLM:
    def __init__(self, companies, year):
        self._e = QueryEntities(companies=companies, fiscal_year=year)
    def with_structured_output(self, schema):
        e = self._e
        class _S:
            def invoke(self, _p):
                if schema is QueryEntities:
                    return e
                return RAGAnswer(answer="Total assets were 5B.", refused=False)
        return _S()


def _deps(tmp_path, companies, year):
    docs = [Document(page_content="cross first total assets five billion",
                     metadata={"source": "cross.pdf", "page": 1, "chunk_id": "cross.pdf::p1::c0"})]
    store = ChromaStore(HashEmbeddings(), str(tmp_path / "chroma"), "ag_test")
    store.add(docs)
    meta = {"cross.pdf": {"company_name": "CrossFirst Bankshares, Inc.",
                          "aliases": ["CrossFirst Bank"], "fiscal_year": "2022"}}
    class _S:
        top_k = 5; bm25_weight = 0.4; vector_weight = 0.6; agentic_max_retries = 1; max_context_tokens = 8000
        refusal_score_threshold = 0.3
    return QueryDeps(store=store, docstore_docs=docs, entity_index=build_index(meta),
                     llm=_LLM(companies, year), settings=_S())


def test_agentic_refuses_unknown_company(tmp_path):
    deps = _deps(tmp_path, ["Nonexistent Corp"], "2022")
    result = answer_agentic("assets of Nonexistent Corp in 2022?", deps)
    assert result.refused is True


def test_agentic_answers_when_resolved(tmp_path):
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2022")
    result = answer_agentic("assets of CrossFirst Bank in 2022?", deps)
    assert result.refused is False
    assert "5B" in result.answer


def test_agentic_records_trace_on_success(tmp_path):
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2022")
    trace = TraceRecorder()
    answer_agentic("assets of CrossFirst Bank in 2022?", deps, trace=trace)

    stages = [s.stage for s in trace.steps]
    assert stages == ["parse_query", "resolve_entities", "retrieve_candidates", "generate"]


def test_agentic_records_rewrite_on_retry(tmp_path, monkeypatch):
    import _deprecated.agentic as agentic_mod

    deps = _deps(tmp_path, ["CrossFirst Bank"], "2022")
    calls = {"n": 0}
    real_retrieve = agentic_mod._retrieve

    def flaky_retrieve(question, mode, sources, deps, trace=None):
        calls["n"] += 1
        if calls["n"] == 1:
            if trace is not None:
                trace.record("retrieve_candidates", candidates=[])
            return []  # first attempt: nothing found -> triggers a retry
        return real_retrieve(question, mode, sources, deps, trace=trace)

    monkeypatch.setattr(agentic_mod, "_retrieve", flaky_retrieve)

    trace = TraceRecorder()
    result = answer_agentic("assets of CrossFirst Bank in 2022?", deps, trace=trace)

    assert result.refused is False
    stages = [s.stage for s in trace.steps]
    assert "rewrite" in stages
    assert stages.count("retrieve_candidates") == 2  # first (empty) + retry
