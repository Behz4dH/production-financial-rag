"""Linear query path: refusal short-circuit + retrieve→generate (mocked LLM)."""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.query import QueryDeps, answer_linear
from rag.generation.schema import RAGAnswer
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
    """Parses to a fixed entity, and generates a fixed answer."""
    def __init__(self, companies, year):
        self._entities = QueryEntities(companies=companies, fiscal_year=year)
    def with_structured_output(self, schema):
        entities, self_ = self._entities, self
        class _S:
            def invoke(self, _p):
                if schema is QueryEntities:
                    return entities
                return RAGAnswer(answer="Total assets were 5B.", refused=False)
        return _S()


def _deps(tmp_path, companies, year):
    docs = [Document(page_content="cross first total assets five billion",
                     metadata={"source": "cross.pdf", "page": 1, "chunk_id": "cross.pdf::p1::c0"})]
    store = ChromaStore(_HashEmbeddings(), str(tmp_path / "chroma"), "q_test")
    store.add(docs)
    meta = {"cross.pdf": {"company_name": "CrossFirst Bankshares, Inc.",
                          "aliases": ["CrossFirst Bank"], "fiscal_year": "2022"}}

    class _S:  # minimal settings stand-in
        top_k = 5; bm25_weight = 0.4; vector_weight = 0.6; max_tokens_per_request = 8000

    return QueryDeps(store=store, docstore_docs=docs, entity_index=build_index(meta),
                     llm=_LLM(companies, year), settings=_S())


def test_answer_refuses_wrong_year(tmp_path):
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2023")
    result = answer_linear("assets of CrossFirst Bank in 2023?", "hybrid", deps)
    assert result.refused is True
    assert result.answer.strip().upper().startswith("N/A")


def test_answer_generates_when_resolved(tmp_path):
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2022")
    result = answer_linear("assets of CrossFirst Bank in 2022?", "hybrid", deps)
    assert result.refused is False
    assert "5B" in result.answer


def test_answer_dispatches_agentic(tmp_path):
    from rag.query import answer
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2022")
    result = answer("assets of CrossFirst Bank in 2022?", "agentic", deps)
    assert result.refused is False
    assert "5B" in result.answer


def test_answer_defaults_to_linear_for_hybrid(tmp_path):
    from rag.query import answer
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2023")
    result = answer("assets of CrossFirst Bank in 2023?", "hybrid", deps)
    assert result.refused is True
