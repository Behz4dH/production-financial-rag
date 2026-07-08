"""Cross-encoder reranking: score each (query, chunk) pair and keep the best.

Simple and transparent — a cross-encoder reads the query and chunk together and
returns a relevance score, which is more precise than the retriever's fusion
ranking for the final few chunks the LLM actually sees.
"""

from langchain_core.documents import Document


def rerank(query: str, docs: list[Document], model, top_n: int) -> list[Document]:
    """Rank by cross-encoder score, keep the top_n, and stamp each kept doc's
    score onto its metadata (query-time only — never persisted) so callers can
    judge how relevant the best match actually was, not just whether any
    match exists."""
    if not docs:
        return docs
    scores = model.predict([(query, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    kept = ranked[:top_n]
    for doc, score in kept:
        doc.metadata["rerank_score"] = float(score)
    return [doc for doc, _ in kept]


def load_reranker(model_name: str):
    """Load the cross-encoder (downloads weights on first use)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)
