"""Query path: resolve entities -> (refuse | retrieve -> generate)."""

from dataclasses import dataclass

from core.config import Settings, get_settings
from rag.generation.generator import generate, refusal
from rag.generation.schema import RAGAnswer
from rag.providers.factory import get_embeddings, get_llm
from rag.retrieval.docstore import load_docstore
from rag.retrieval.entity_resolver import build_index, parse_query, resolve
from rag.retrieval.hybrid import build_hybrid_retriever, vector_only_retriever
from rag.retrieval.reranker import load_reranker, rerank
from rag.retrieval.store import ChromaStore


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
    return QueryDeps(store=store, docstore_docs=docstore_docs,
                     entity_index=build_index(meta), llm=get_llm(settings),
                     settings=settings, reranker=load_reranker(settings.reranker_model))


def _retrieve(question, mode, sources, deps):
    s = deps.settings
    if mode == "basic":
        retriever = vector_only_retriever(deps.store, sources, s.top_k)
    else:  # hybrid
        retriever = build_hybrid_retriever(deps.store, deps.docstore_docs, sources,
                                           s.top_k, s.bm25_weight, s.vector_weight)
    candidates = retriever.invoke(question)
    if deps.reranker is None:
        return candidates
    return rerank(question, candidates, deps.reranker, s.top_n)


def relevance_refusal_reason(docs: list, threshold: float) -> str | None:
    """None if docs clear the bar; otherwise why they don't.

    top_n retrieval almost always returns *something*, even when nothing is
    actually relevant, so an empty list is rare. The cross-encoder's
    rerank_score (stamped on docs by rag.retrieval.reranker.rerank) is the
    real signal: it's calibrated per-query, unlike raw retriever fusion
    scores. Without a reranker (e.g. in tests) there's no such score to judge
    by, so we only fall back to the plain "nothing came back" check.
    """
    if not docs:
        return "no relevant excerpts retrieved"
    top_score = docs[0].metadata.get("rerank_score")
    if top_score is not None and top_score < threshold:
        return f"top relevance score {top_score:.2f} below threshold {threshold}"
    return None


def answer_linear(question: str, mode: str, deps: QueryDeps) -> RAGAnswer:
    entities = parse_query(question, deps.llm)
    res = resolve(entities, deps.entity_index)
    if res.refuse:
        return refusal(f"no filing matches {res.unresolved}")
    docs = _retrieve(question, mode, res.sources, deps)
    reason = relevance_refusal_reason(docs, deps.settings.refusal_score_threshold)
    if reason:
        return refusal(reason)
    return generate(question, docs, deps.llm, deps.settings.max_context_tokens)


def answer(question: str, mode: str, deps: QueryDeps) -> RAGAnswer:
    if mode == "agentic":
        from rag.agentic import answer_agentic
        return answer_agentic(question, deps)
    return answer_linear(question, mode, deps)
