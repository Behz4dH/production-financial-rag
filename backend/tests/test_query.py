"""Linear query path: refusal short-circuit + retrieve→generate (mocked LLM)."""

from langchain_core.documents import Document

from rag.query import QueryDeps, _retrieve, answer_linear, relevance_refusal_reason
from rag.generation.schema import RAGAnswer
from rag.retrieval.entity_resolver import QueryEntities, build_index
from rag.retrieval.store import ChromaStore
from rag.trace import TraceRecorder
from tests.fakes import HashEmbeddings


class _LLM:
    """Parses to a fixed entity, and generates a fixed (or configurably
    refused, with reasoning) answer."""
    def __init__(self, companies, year, answer="Total assets were 5B.", refused=False, reasoning=""):
        self._entities = QueryEntities(companies=companies, fiscal_year=year)
        self._answer, self._refused, self._reasoning = answer, refused, reasoning
    def with_structured_output(self, schema):
        entities, answer, refused, reasoning = self._entities, self._answer, self._refused, self._reasoning
        class _S:
            def invoke(self, _p):
                if schema is QueryEntities:
                    return entities
                return RAGAnswer(answer=answer, refused=refused, reasoning=reasoning)
        return _S()


def _deps(tmp_path, companies, year, **llm_kwargs):
    docs = [Document(page_content="cross first total assets five billion",
                     metadata={"source": "cross.pdf", "page": 1, "chunk_id": "cross.pdf::p1::c0"})]
    store = ChromaStore(HashEmbeddings(), str(tmp_path / "chroma"), "q_test")
    store.add(docs)
    meta = {"cross.pdf": {"company_name": "CrossFirst Bankshares, Inc.",
                          "aliases": ["CrossFirst Bank"], "fiscal_year": "2022"}}

    class _S:  # minimal settings stand-in
        top_k = 5; bm25_weight = 0.4; vector_weight = 0.6; max_context_tokens = 8000
        refusal_score_threshold = 0.3

    return QueryDeps(store=store, docstore_docs=docs, entity_index=build_index(meta),
                     llm=_LLM(companies, year, **llm_kwargs), settings=_S())


def test_answer_reaches_generation_despite_wrong_year(tmp_path):
    # entity resolution no longer gates on fiscal year -- a filing may
    # legitimately contain comparatives for the asked year, so the question
    # still reaches generation. This fake LLM answers regardless of input,
    # proving resolution didn't short-circuit it the way it used to.
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2023")
    result = answer_linear("assets of CrossFirst Bank in 2023?", "hybrid", deps)
    assert result.refused is False
    assert "5B" in result.answer


def test_generation_refusal_carries_the_models_own_reasoning(tmp_path):
    # this is the mechanism that replaces the old hard year gate: the model
    # sees the actual excerpt and has to say *why* it doesn't answer the
    # question, not just a fixed "insufficient" string.
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2023", answer="N/A",
                refused=True, reasoning="the excerpt covers FY2022, not the asked FY2023")
    result = answer_linear("assets of CrossFirst Bank in 2023?", "hybrid", deps)
    assert result.refused is True
    assert result.reason == "the excerpt covers FY2022, not the asked FY2023"
    assert result.reasoning == result.reason


def test_relevance_refusal_reason_empty_docs():
    assert relevance_refusal_reason([], threshold=0.3) == "no relevant excerpts retrieved"


def test_relevance_refusal_reason_below_threshold():
    doc = Document(page_content="x", metadata={"rerank_score": 0.1})
    reason = relevance_refusal_reason([doc], threshold=0.3)
    assert reason is not None
    assert "0.10" in reason and "0.3" in reason


def test_relevance_refusal_reason_above_threshold():
    doc = Document(page_content="x", metadata={"rerank_score": 0.9})
    assert relevance_refusal_reason([doc], threshold=0.3) is None


def test_relevance_refusal_reason_no_score_defers_to_presence():
    # no reranker ran (metadata has no rerank_score) -> presence alone is enough
    doc = Document(page_content="x", metadata={})
    assert relevance_refusal_reason([doc], threshold=0.3) is None


def test_answer_generates_when_resolved(tmp_path):
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2022")
    result = answer_linear("assets of CrossFirst Bank in 2022?", "hybrid", deps)
    assert result.refused is False
    assert "5B" in result.answer


def test_answer_dispatches_to_answer_linear(tmp_path):
    from rag.query import answer
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2022")
    result = answer("assets of CrossFirst Bank in 2022?", "hybrid", deps)
    assert result.refused is False
    assert "5B" in result.answer


def test_answer_defaults_to_linear_for_hybrid(tmp_path):
    from rag.query import answer
    deps = _deps(tmp_path, ["Nonexistent Corp"], "2022")
    result = answer("assets of Nonexistent Corp in 2022?", "hybrid", deps)
    assert result.refused is True


def test_answer_linear_records_trace_on_success(tmp_path):
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2022")
    trace = TraceRecorder()
    answer_linear("assets of CrossFirst Bank in 2022?", "hybrid", deps, trace=trace)

    stages = [s.stage for s in trace.steps]
    assert stages == ["parse_query", "resolve_entities", "retrieve_candidates",
                      "expand_pages", "generate"]
    assert trace.steps[0].data["companies"] == ["CrossFirst Bank"]
    assert trace.steps[-1].data["answer"] == "Total assets were 5B."


def test_answer_linear_records_trace_on_entity_refusal(tmp_path):
    deps = _deps(tmp_path, ["Nonexistent Corp"], "2022")
    trace = TraceRecorder()
    answer_linear("assets of Nonexistent Corp in 2022?", "hybrid", deps, trace=trace)

    stages = [s.stage for s in trace.steps]
    assert stages == ["parse_query", "resolve_entities", "refuse"]
    assert "Nonexistent Corp" in trace.steps[-1].data["reason"]


def test_answer_without_trace_is_unaffected(tmp_path):
    # trace defaults to None — this must behave exactly as before (no crash, no extra work).
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2022")
    result = answer_linear("assets of CrossFirst Bank in 2022?", "hybrid", deps)
    assert result.refused is False


# --- Multi-source ("compare A vs B") retrieval decomposition ---

class _FakeCrossEncoder:
    """High score for strong.pdf's chunk, low (sub-threshold) for weak.pdf's."""
    def predict(self, pairs):
        return [0.9 if "total shareholders" in doc else 0.1 for _q, doc in pairs]


def _deps_crowd(tmp_path, reranker=None):
    docs = [
        Document(page_content="total shareholders equity",
                 metadata={"source": "strong.pdf", "page": 1, "chunk_id": "strong.pdf::p1::c0"}),
        Document(page_content="equity mentioned briefly in a footnote somewhere",
                 metadata={"source": "weak.pdf", "page": 1, "chunk_id": "weak.pdf::p1::c0"}),
    ]
    store = ChromaStore(HashEmbeddings(), str(tmp_path / "chroma"), "q_test_crowd")
    store.add(docs)
    meta = {"strong.pdf": {"company_name": "Strong Co", "aliases": [], "fiscal_year": "2022"},
            "weak.pdf": {"company_name": "Weak Co", "aliases": [], "fiscal_year": "2022"}}

    class _S:  # top_k=top_n=1: tight enough that a *shared* budget would starve one source
        top_k = 1; bm25_weight = 0.4; vector_weight = 0.6; max_context_tokens = 8000
        refusal_score_threshold = 0.3; top_n = 1

    return QueryDeps(store=store, docstore_docs=docs, entity_index=build_index(meta),
                     llm=_LLM(["Strong Co", "Weak Co"], "2022"), settings=_S(), reranker=reranker)


def test_retrieve_one_documents_the_shared_budget_bug(tmp_path):
    # Documents the bug being fixed: a single combined call at this tight
    # top_k crowds "weak.pdf" out entirely, even though it's a real resolved
    # source. `_retrieve` (below) must not exhibit this.
    from rag.query import _retrieve_one
    deps = _deps_crowd(tmp_path)
    docs = _retrieve_one("total shareholders equity", "basic", ["strong.pdf", "weak.pdf"], deps)
    assert {d.metadata["source"] for d in docs} == {"strong.pdf"}


def test_retrieve_gives_each_resolved_source_its_own_budget(tmp_path):
    deps = _deps_crowd(tmp_path)
    docs = _retrieve("total shareholders equity", "basic", ["strong.pdf", "weak.pdf"], deps)
    assert {d.metadata["source"] for d in docs} == {"strong.pdf", "weak.pdf"}


def test_answer_linear_grounds_both_companies_in_a_compare_question(tmp_path):
    deps = _deps_crowd(tmp_path)
    trace = TraceRecorder()
    result = answer_linear(
        "Which company had higher equity: Strong Co or Weak Co, in 2022?", "basic", deps, trace=trace)

    assert result.refused is False
    stages = [s.stage for s in trace.steps]
    assert stages.count("retrieve_candidates") == 2  # one call per resolved source
    retrieved_sources = {s.data.get("source") for s in trace.steps if s.stage == "retrieve_candidates"}
    assert retrieved_sources == {"strong.pdf", "weak.pdf"}
    assert trace.steps[-1].data["context_chunks"] == 2  # both companies' chunks reached generation


def test_answer_linear_refuses_compare_when_one_company_is_weak(tmp_path):
    deps = _deps_crowd(tmp_path, reranker=_FakeCrossEncoder())
    result = answer_linear(
        "Which company had higher equity: Strong Co or Weak Co, in 2022?", "basic", deps)
    assert result.refused is True
    assert "weak.pdf" in result.reason


def test_relevance_refusal_reason_multi_source_missing_one_source():
    doc = Document(page_content="x", metadata={"source": "a.pdf", "rerank_score": 0.9})
    reason = relevance_refusal_reason([doc], threshold=0.3, sources=["a.pdf", "b.pdf"])
    assert reason is not None and "b.pdf" in reason


def test_relevance_refusal_reason_multi_source_both_strong():
    doc_a = Document(page_content="x", metadata={"source": "a.pdf", "rerank_score": 0.9})
    doc_b = Document(page_content="y", metadata={"source": "b.pdf", "rerank_score": 0.8})
    assert relevance_refusal_reason([doc_a, doc_b], threshold=0.3, sources=["a.pdf", "b.pdf"]) is None


def test_relevance_refusal_reason_multi_source_no_score_defers_to_presence():
    doc_a = Document(page_content="x", metadata={"source": "a.pdf"})
    doc_b = Document(page_content="y", metadata={"source": "b.pdf"})
    assert relevance_refusal_reason([doc_a, doc_b], threshold=0.3, sources=["a.pdf", "b.pdf"]) is None


def test_relevance_refusal_reason_single_source_list_is_byte_identical_to_no_sources_arg():
    # sources with len<=1 must behave exactly like the old 2-arg call (single-
    # source phrasing, not the per-source "source: ..." phrasing).
    doc = Document(page_content="x", metadata={"source": "a.pdf", "rerank_score": 0.1})
    reason = relevance_refusal_reason([doc], threshold=0.3, sources=["a.pdf"])
    assert reason == "top relevance score 0.10 below threshold 0.3"


def _deps_sort(tmp_path):
    docs = [
        Document(page_content="A1 HIGH doc about equity",
                 metadata={"source": "a.pdf", "page": 1, "chunk_id": "a.pdf::p1::c0"}),
        Document(page_content="A2 LOW doc about equity",
                 metadata={"source": "a.pdf", "page": 2, "chunk_id": "a.pdf::p2::c0"}),
        Document(page_content="B1 HIGH doc about equity",
                 metadata={"source": "b.pdf", "page": 1, "chunk_id": "b.pdf::p1::c0"}),
        Document(page_content="B2 LOW doc about equity",
                 metadata={"source": "b.pdf", "page": 2, "chunk_id": "b.pdf::p2::c0"}),
    ]
    store = ChromaStore(HashEmbeddings(), str(tmp_path / "chroma"), "q_test_sort")
    store.add(docs)

    class _RankedFakeCrossEncoder:
        def predict(self, pairs):
            return [0.9 if "A1" in d else 0.85 if "B1" in d else 0.3 if "A2" in d else 0.2
                   for _q, d in pairs]

    class _S:
        top_k = 5; bm25_weight = 0.4; vector_weight = 0.6; max_context_tokens = 8000
        refusal_score_threshold = 0.3; top_n = 2

    return QueryDeps(store=store, docstore_docs=docs, entity_index=build_index({}),
                     llm=None, settings=_S(), reranker=_RankedFakeCrossEncoder())


def test_retrieve_merges_multi_source_results_globally_best_first(tmp_path):
    deps = _deps_sort(tmp_path)
    docs = _retrieve("equity", "basic", ["a.pdf", "b.pdf"], deps)
    scores = [d.metadata["rerank_score"] for d in docs]
    assert scores == sorted(scores, reverse=True)  # globally sorted, not per-source blocks
    assert [d.metadata["source"] for d in docs] == ["a.pdf", "b.pdf", "a.pdf", "b.pdf"]  # interleaved


def test_build_deps_builds_reranker_with_its_own_scoring_model(monkeypatch, tmp_path):
    """The reranker's LLM is settings.reranker_llm_model (70b), not the primary
    generation model — its score gates refusal, and the 8b primary degenerates
    to all-zero scores on batches of weak candidates."""
    import rag.query as query
    from core.config import Settings

    llm_calls = []

    def fake_get_llm(settings, model=None):
        llm_calls.append(model)
        return f"llm:{model or settings.primary_model}"

    captured = {}

    def fake_build_reranker(settings, llm=None):
        captured["llm"] = llm
        return "reranker"

    monkeypatch.setattr(query, "get_llm", fake_get_llm)
    monkeypatch.setattr(query, "get_embeddings", lambda s: HashEmbeddings())
    monkeypatch.setattr(query, "build_reranker", fake_build_reranker)

    settings = Settings(groq_api_key="test-key", _env_file=None,
                        chroma_dir=str(tmp_path / "chroma"),
                        docstore_path=str(tmp_path / "docstore.jsonl"),
                        doc_metadata_path=str(tmp_path / "meta.json"))
    deps = query.build_deps(settings)

    assert deps.llm == f"llm:{settings.primary_model}"
    assert captured["llm"] == f"llm:{settings.reranker_llm_model}"


def _pagedoc(source, page, cid, text, score=None):
    md = {"source": source, "page": page, "chunk_id": f"{source}::p{page}::{cid}"}
    if score is not None:
        md["rerank_score"] = score
    return Document(page_content=text, metadata=md)


def test_expand_to_pages_merges_page_chunks_and_keeps_best_first_order():
    """Kept chunks expand to their FULL page (all docstore chunks of that
    source+page) so generation sees labels/units/headers, not an isolated
    fragment. Page order inherits the kept docs' best-first order."""
    from rag.query import _expand_to_pages

    docstore = [
        _pagedoc("a.pdf", 1, "c0", "Income statement header US$ million 2022 2021"),
        _pagedoc("a.pdf", 1, "t0", "Profit for the Year -- 88.1; 196.6"),
        _pagedoc("a.pdf", 2, "c0", "Unrelated page two text"),
        _pagedoc("b.pdf", 9, "c0", "Balance sheet CHF 000"),
        _pagedoc("b.pdf", 9, "t3", "Total equity -- 146,469"),
    ]
    kept = [  # best-first from rerank; only ONE chunk per page was kept
        _pagedoc("b.pdf", 9, "t3", "Total equity -- 146,469", score=0.9),
        _pagedoc("a.pdf", 1, "t0", "Profit for the Year -- 88.1; 196.6", score=0.7),
    ]
    pages = _expand_to_pages(kept, docstore)

    assert [(d.metadata["source"], d.metadata["page"]) for d in pages] == [("b.pdf", 9), ("a.pdf", 1)]
    assert "Balance sheet CHF 000" in pages[0].page_content        # sibling chunk pulled in
    assert "Income statement header" in pages[1].page_content      # labels now visible
    assert "Unrelated page two" not in "".join(p.page_content for p in pages)
    assert pages[0].metadata["rerank_score"] == 0.9  # best member score carried


def test_expand_to_pages_dedupes_multiple_kept_chunks_from_same_page():
    from rag.query import _expand_to_pages

    docstore = [_pagedoc("a.pdf", 1, "c0", "alpha"), _pagedoc("a.pdf", 1, "c1", "beta")]
    kept = [_pagedoc("a.pdf", 1, "c0", "alpha", score=0.8),
            _pagedoc("a.pdf", 1, "c1", "beta", score=0.6)]
    pages = _expand_to_pages(kept, docstore)
    assert len(pages) == 1
    assert "alpha" in pages[0].page_content and "beta" in pages[0].page_content


def test_answer_linear_generates_from_expanded_pages(tmp_path, monkeypatch):
    import rag.query as query

    deps = _deps(tmp_path, ["CrossFirst Bankshares"], "2022")
    # a sibling chunk on the same page that retrieval never returned
    deps.docstore_docs.append(_pagedoc("cross.pdf", 1, "c9", "UNIT NOTE: in thousands"))

    captured = {}
    real_generate = query.generate

    def spy_generate(question, docs, llm, max_tokens=None, multi_source=False):
        captured["docs"] = docs
        return real_generate(question, docs, llm, max_tokens, multi_source=multi_source)

    monkeypatch.setattr(query, "generate", spy_generate)
    query.answer_linear("What were total assets?", "basic", deps)

    assert len(captured["docs"]) >= 1
    joined = " ".join(d.page_content for d in captured["docs"])
    assert "UNIT NOTE: in thousands" in joined  # generation saw the whole page


def test_answer_linear_stamps_company_names_on_expanded_pages(tmp_path, monkeypatch):
    """Query-time company annotation (from the entity index — NOT persisted
    onto chunks) so compare-question generation can attribute excerpts."""
    import rag.query as query

    deps = _deps(tmp_path, ["CrossFirst Bankshares"], "2022")
    captured = {}
    real_generate = query.generate

    def spy_generate(question, docs, llm, max_tokens=None, multi_source=False):
        captured["docs"] = docs
        return real_generate(question, docs, llm, max_tokens, multi_source=multi_source)

    monkeypatch.setattr(query, "generate", spy_generate)
    query.answer_linear("What were total assets?", "basic", deps)

    assert captured["docs"][0].metadata.get("company") == "CrossFirst Bankshares, Inc."


def test_resolved_source_missing_from_index_refuses_instead_of_crashing(tmp_path):
    """Metadata is a cache and may know filings the current index doesn't
    (e.g. the sliced corpus). That's a refusal with a precise reason — not an
    unhandled ValueError from BM25Retriever.from_documents([])."""
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2022")
    deps.docstore_docs.clear()  # index has nothing for the resolved source

    result = answer_linear("assets of CrossFirst Bank in 2022?", "hybrid", deps)
    assert result.refused is True
    assert "not in the current index" in result.reason
    assert "cross.pdf" in result.reason


def test_corpus_inventory_question_answers_from_catalog(tmp_path):
    """'Which companies do we have?' has no entity to resolve — it must route
    to a deterministic catalog answer (doc_metadata ∩ indexed sources), not
    die at the entity gate."""
    from rag.retrieval.entity_resolver import QueryEntities

    deps = _deps(tmp_path, [], None)
    deps.llm._entities = QueryEntities(companies=[], fiscal_year=None,
                                       asks_corpus_inventory=True)
    # a metadata-known but UN-indexed filing must not be listed
    deps.entity_index.append({"source": "parked.pdf", "name": "Parked Corp",
                              "fiscal_year": "2022"})

    trace = TraceRecorder()
    result = answer_linear("what companies do we have annual data from?",
                           "hybrid", deps, trace=trace)
    assert result.refused is False
    assert "CrossFirst Bankshares, Inc." in result.answer
    assert "Parked Corp" not in result.answer
    assert "corpus_inventory" in [s.stage for s in trace.steps]
