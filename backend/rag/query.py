"""Query path: resolve entities -> (refuse | retrieve -> generate)."""

from dataclasses import dataclass

from langchain_core.documents import Document

from core.config import Settings, get_settings
from rag.generation.generator import generate, refusal
from rag.generation.schema import RAGAnswer
from rag.providers.factory import get_embeddings, get_llm
from rag.retrieval.docstore import load_docstore
from rag.retrieval.entity_resolver import build_index, parse_query, resolve
from rag.retrieval.hybrid import build_hybrid_retriever, vector_only_retriever
from rag.retrieval.reranker import build_reranker, rerank
from rag.retrieval.store import ChromaStore
from rag.trace import TraceRecorder


@dataclass
class QueryDeps:
    store: ChromaStore
    docstore_docs: list
    entity_index: list
    llm: object
    settings: object
    reranker: object = None  # cross-encoder; None skips reranking (e.g. in tests)


def build_deps(settings: Settings | None = None) -> QueryDeps:
    import json
    from pathlib import Path

    settings = settings or get_settings()
    store = ChromaStore(get_embeddings(settings), settings.chroma_dir, settings.collection_name)
    docstore_docs = load_docstore(settings.docstore_path)
    meta_path = Path(settings.doc_metadata_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    llm = get_llm(settings)
    # The reranker scores with its OWN model (settings.reranker_llm_model,
    # the 70b) — its score gates refusal, and the 8b primary degenerates to
    # all-zero scores on batches with no obviously-relevant snippet.
    reranker_llm = get_llm(settings, settings.reranker_llm_model)
    return QueryDeps(store=store, docstore_docs=docstore_docs,
                     entity_index=build_index(meta), llm=llm,
                     settings=settings, reranker=build_reranker(settings, reranker_llm))


def _doc_snippet(d, with_score: bool = False) -> dict:
    out = {"source": d.metadata.get("source", ""), "page": d.metadata.get("page"),
           "snippet": d.page_content[:200]}
    if with_score:
        out["score"] = d.metadata.get("rerank_score")
    return out


def _retrieve_one(question, mode, sources, deps, trace: TraceRecorder | None = None,
                  source_label: str | None = None):
    """Retrieve + (optionally) rerank against exactly one retrieval call.

    `source_label` only affects trace annotation: it's None for the ordinary
    single-source dispatch below (so that path's trace shape is byte-for-byte
    unchanged), and set to the one source each decomposed call in `_retrieve`
    is scoped to, so the trace/demo dashboard can tell per-company retrieval
    steps apart.
    """
    s = deps.settings
    if mode == "basic":
        retriever = vector_only_retriever(deps.store, sources, s.top_k)
    else:  # hybrid
        retriever = build_hybrid_retriever(deps.store, deps.docstore_docs, sources,
                                           s.top_k, s.bm25_weight, s.vector_weight)
    candidates = retriever.invoke(question)
    if trace is not None:
        extra = {"source": source_label} if source_label is not None else {}
        trace.record("retrieve_candidates", candidates=[_doc_snippet(d) for d in candidates], **extra)
    if deps.reranker is None:
        return candidates
    reranked = rerank(question, candidates, deps.reranker, s.top_n)
    if trace is not None:
        extra = {"source": source_label} if source_label is not None else {}
        trace.record("rerank", chunks=[_doc_snippet(d, with_score=True) for d in reranked], **extra)
    return reranked


def _retrieve(question, mode, sources, deps, trace: TraceRecorder | None = None):
    """Retrieve candidates for the resolved source(s).

    A single source (the overwhelmingly common case) is one retrieval call,
    unchanged from before. Multiple sources (a "compare A vs B" question)
    each get their OWN top_k/top_n budget instead of one shared budget over
    the combined candidate pool -- otherwise a company whose chunks score
    lower across the board gets crowded out of the results entirely by the
    other company's stronger matches.

    Per-source results are merged into one best-first list (sorted by
    rerank_score): with a reranker this makes the list globally best-first
    (not just best-first within each source's block), which is what
    generator._fit_context's token-budget trimming assumes. Without a
    reranker there's no score to sort by (every doc's score is equally
    absent), so the sort is a no-op and the list stays in source-resolution
    order -- the same "no ordering to pretend to have" behavior the
    single-source path already has today.
    """
    if len(sources) <= 1:
        return _retrieve_one(question, mode, sources, deps, trace=trace)
    per_source = [_retrieve_one(question, mode, [source], deps, trace=trace, source_label=source)
                 for source in sources]
    docs = [d for group in per_source for d in group]
    docs.sort(key=lambda d: d.metadata.get("rerank_score", 0.0), reverse=True)
    return docs


def _companies_by_source(entity_index: list) -> dict[str, str]:
    """source -> primary company name (build_index appends company_name first)."""
    out: dict[str, str] = {}
    for rec in entity_index:
        out.setdefault(rec["source"], rec["name"])
    return out


def _corpus_inventory_answer(deps: QueryDeps, trace: TraceRecorder | None = None) -> RAGAnswer:
    """Deterministic catalog answer for questions about the corpus itself.

    'Which companies do we have?' has no entity to resolve, so it can't go
    through retrieval — and it doesn't need to: the answer IS the metadata
    catalog, restricted to sources actually present in the index (metadata is
    a cache and may know parked filings). No LLM, nothing to hallucinate.
    """
    indexed = {d.metadata.get("source") for d in deps.docstore_docs}
    companies = _companies_by_source(deps.entity_index)
    years = {rec["source"]: rec.get("fiscal_year") for rec in deps.entity_index}
    lines = sorted(f"- {name} (fiscal year {years.get(src) or 'unknown'})"
                   for src, name in companies.items() if src in indexed)
    answer_text = (f"The current index contains {len(lines)} annual filings:\n"
                   + "\n".join(lines))
    if trace is not None:
        trace.record("corpus_inventory", filings=len(lines))
    return RAGAnswer(
        answer=answer_text, refused=False, confidence="high",
        reasoning="Answered deterministically from the ingested document catalog "
                  "(doc_metadata.json restricted to indexed sources) — no retrieval "
                  "or generation involved.")


def _expand_to_pages(docs: list, docstore_docs: list) -> list:
    """Expand kept chunks to their full page for generation — retrieve small,
    generate big.

    A reranked chunk is term-dense enough to *find*, but the page's labeling —
    column headers, units notes ("US$ million"), section titles — lives in
    sibling chunks, and generation's strict matching then refuses correctly-
    but-needlessly ("88.1 is a possible match but the year can't be
    confirmed"). One Document per unique (source, page), holding every
    docstore chunk of that page; page order inherits the kept docs'
    best-first order, and each page carries its best member's rerank_score.
    Citations are unaffected: they were already derived from (source, page).
    """
    by_page: dict[tuple, list] = {}
    for d in docstore_docs:
        by_page.setdefault((d.metadata.get("source"), d.metadata.get("page")), []).append(d)
    out: list[Document] = []
    seen: set[tuple] = set()
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        members = sorted(by_page.get(key, [d]),
                         key=lambda m: m.metadata.get("chunk_id", ""))
        scores = [k.metadata.get("rerank_score") for k in docs
                  if (k.metadata.get("source"), k.metadata.get("page")) == key]
        md: dict = {"source": key[0], "page": key[1]}
        best = max((s for s in scores if s is not None), default=None)
        if best is not None:
            md["rerank_score"] = best
        out.append(Document(page_content="\n".join(m.page_content for m in members),
                            metadata=md))
    return out


def relevance_refusal_reason(docs: list, threshold: float, sources: list[str] | None = None) -> str | None:
    """None if docs clear the bar; otherwise why they don't.

    top_n retrieval almost always returns *something*, even when nothing is
    actually relevant, so an empty list is rare. The cross-encoder's
    rerank_score (stamped on docs by rag.retrieval.reranker.rerank) is the
    real signal: it's calibrated per-query, unlike raw retriever fusion
    scores. Without a reranker (e.g. in tests) there's no such score to judge
    by, so we only fall back to the plain "nothing came back" check.

    A multi-source question (e.g. "compare A vs B") needs every resolved
    source individually grounded -- one company's strong top score can
    otherwise hide another company's weak or entirely missing retrieval,
    producing a one-sided "comparison" instead of a refusal. `sources` is
    the full set of resolved source filenames for the question (not derived
    from `docs`, since a source that returned zero docs wouldn't show up in
    `docs` at all); with 0 or 1 sources this is identical to the single-score
    check below, which is the vast majority of calls.
    """
    if not docs:
        return "no relevant excerpts retrieved"
    if sources is not None and len(sources) > 1:
        return _multi_source_refusal_reason(docs, threshold, sources)
    top_score = docs[0].metadata.get("rerank_score")
    if top_score is not None and top_score < threshold:
        return f"top relevance score {top_score:.2f} below threshold {threshold}"
    return None


def _multi_source_refusal_reason(docs: list, threshold: float, sources: list[str]) -> str | None:
    by_source: dict[str, list] = {source: [] for source in sources}
    for d in docs:
        source = d.metadata.get("source")
        if source in by_source:
            by_source[source].append(d)
    for source in sources:
        source_docs = by_source[source]
        if not source_docs:
            return f"no relevant excerpts retrieved for {source}"
        scores = [d.metadata.get("rerank_score") for d in source_docs]
        if all(score is None for score in scores):
            continue  # no reranker ran; presence alone is enough for this source
        top_score = max(score for score in scores if score is not None)
        if top_score < threshold:
            return f"{source}: top relevance score {top_score:.2f} below threshold {threshold}"
    return None


def answer_linear(question: str, mode: str, deps: QueryDeps,
                  trace: TraceRecorder | None = None) -> RAGAnswer:
    entities = parse_query(question, deps.llm)
    if trace is not None:
        trace.record("parse_query", companies=entities.companies, fiscal_year=entities.fiscal_year,
                     asks_corpus_inventory=entities.asks_corpus_inventory)
    if entities.asks_corpus_inventory:
        return _corpus_inventory_answer(deps, trace=trace)
    res = resolve(entities, deps.entity_index)
    if trace is not None:
        trace.record("resolve_entities", sources=res.sources, unresolved=res.unresolved)
    if res.refuse:
        reason = f"no filing matches {res.unresolved}"
        if trace is not None:
            trace.record("refuse", reason=reason)
        return refusal(reason)
    # Metadata is a cache and may know filings the index doesn't (sliced
    # corpus, failed ingest). A resolved-but-unindexed source is a refusal
    # with a precise reason — not an unhandled crash in BM25 construction.
    indexed = {d.metadata.get("source") for d in deps.docstore_docs}
    missing = [s for s in res.sources if s not in indexed]
    if missing:
        reason = (f"filing(s) {missing} matched the asked company but are "
                  "not in the current index — restore the document(s) and re-ingest")
        if trace is not None:
            trace.record("refuse", reason=reason)
        return refusal(reason)
    docs = _retrieve(question, mode, res.sources, deps, trace=trace)
    reason = relevance_refusal_reason(docs, deps.settings.refusal_score_threshold, sources=res.sources)
    if reason:
        if trace is not None:
            trace.record("refuse", reason=reason)
        return refusal(reason)
    # The refusal gate judged the precise chunks; generation gets their full
    # pages so labels/units/headers are visible (retrieve small, generate big).
    docs = _expand_to_pages(docs, deps.docstore_docs)
    # Query-time company annotation (from the entity index, NEVER persisted
    # onto chunks): sources are opaque hash filenames, and generation must be
    # able to attribute excerpts to companies — essential on compares.
    companies = _companies_by_source(deps.entity_index)
    for d in docs:
        name = companies.get(d.metadata.get("source"))
        if name:
            d.metadata["company"] = name
    if trace is not None:
        trace.record("expand_pages",
                     pages=[{"source": d.metadata.get("source"), "page": d.metadata.get("page")}
                            for d in docs])
    result = generate(question, docs, deps.llm, deps.settings.max_context_tokens,
                      multi_source=len(res.sources) > 1)
    if trace is not None:
        trace.record("generate", context_chunks=len(docs), refused=result.refused,
                     confidence=result.confidence, answer=result.answer, reasoning=result.reasoning)
    return result


def answer(question: str, mode: str, deps: QueryDeps,
          trace: TraceRecorder | None = None) -> RAGAnswer:
    return answer_linear(question, mode, deps, trace=trace)
