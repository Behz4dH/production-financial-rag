"""Agentic path: refusal on unknown entity; answer when resolved (mocked LLM)."""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.agentic import answer_agentic
from rag.generation.schema import RAGAnswer
from rag.query import QueryDeps
from rag.retrieval.entity_resolver import QueryEntities, build_index
from rag.retrieval.store import ChromaStore


class _HashEmbeddings(Embeddings):
    def _v(self, t):
        v = [0.0] * 8
        for tok in t.lower().split():
            v[hash(tok) % 8] += 1.0
        return v
    def embed_documents(self, texts): return [self._v(t) for t in texts]
    def embed_query(self, text): return self._v(text)


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
    store = ChromaStore(_HashEmbeddings(), str(tmp_path / "chroma"), "ag_test")
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
