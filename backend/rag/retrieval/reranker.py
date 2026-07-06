"""Cross-encoder reranking: score each (query, chunk) pair and keep the best.

Simple and transparent — a cross-encoder reads the query and chunk together and
returns a relevance score, which is more precise than the retriever's fusion
ranking for the final few chunks the LLM actually sees.
"""

from langchain_core.documents import Document


def rerank(query: str, docs: list[Document], model, top_n: int) -> list[Document]:
    if not docs:
        return docs
    scores = model.predict([(query, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]


def load_reranker(model_name: str):
    """Load the cross-encoder (downloads weights on first use)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)
