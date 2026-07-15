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
    reranker: object = None  # .predict(pairs) scorer; None skips reranking (tests)


def build_deps(settings: Settings | None = None) -> QueryDeps:
    import json
    from pathlib import Path

    settings = settings or get_settings()
    store = ChromaStore(get_embeddings(settings), settings.chroma_dir, settings.collection_name)
    docstore_docs = load_docstore(settings.docstore_path)
    meta_path = Path(settings.doc_metadata_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    llm = get_llm(settings)
    # The reranker gets its own (stronger) model: its score gates refusal.
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
    """Retrieve and (optionally) rerank for one retrieval call.

    `source_label` annotates the trace on decomposed multi-source calls so
    per-company steps are distinguishable; None on the single-source path.
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

    Multi-source questions get a per-source top_k/top_n budget — a shared
    budget lets one company's stronger matches crowd the other out entirely —
    then merge globally best-first by rerank_score (a no-op without a
    reranker, where no scores exist).
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
    """Deterministic catalog answer for questions about the corpus itself,
    restricted to sources actually present in the index (the metadata file is
    a cache and may know filings that aren't ingested). No LLM involved."""
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
    """Expand kept chunks to their full pages for generation.

    Retrieve small, generate big: the chunk is the retrieval unit, but the
    page carries the labeling (column headers, units notes) that strict-match
    generation needs. One Document per unique (source, page) with every
    docstore chunk of that page; page order inherits the kept docs'
    best-first order, and each page carries its best member's rerank_score.
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
    """None if docs clear the relevance bar; otherwise the refusal reason.

    The rerank_score is the signal; without a reranker only the empty-result
    check applies. Multi-source questions require every resolved source
    individually grounded — one company's strong score must not hide
    another's missing retrieval. `sources` comes from resolution, not from
    `docs`: a source that returned nothing wouldn't appear in `docs` at all.
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
    # A resolved source may be absent from the index (metadata is a cache);
    # refuse with a reason rather than crash building BM25 over zero docs.
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
    # The refusal gate judged the precise chunks; generation gets full pages.
    docs = _expand_to_pages(docs, deps.docstore_docs)
    # Company annotation happens at query time only — entity attributes are
    # never persisted onto chunks. Sources are opaque hash filenames, and
    # generation must attribute excerpts to companies on compare questions.
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
