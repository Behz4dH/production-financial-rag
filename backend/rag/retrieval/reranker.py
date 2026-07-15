"""Final-stage reranking: score (query, chunk) pairs, keep the best.

Both rerankers expose ``.predict(pairs) -> list[float]``, so the query path
is agnostic to which is configured. The LLM reranker's rubric scores whether
a chunk contains the asked-for figure — the signal the refusal gate needs —
rather than topical similarity.
"""

import json
import re

from langchain_core.documents import Document

_RERANK_PROMPT = """You are ranking retrieved excerpts from a company's annual \
report by how useful each is for answering the question. An excerpt is highly \
relevant (score near 1.0) only if it contains the specific figure or fact the \
question asks for, for the right period. Excerpts that merely mention related \
words score low. Score EVERY excerpt id exactly once.

Question: {question}

Excerpts:
{excerpts}

Respond with ONLY a JSON array, no prose, one entry per excerpt:
[{{"id": 0, "score": 0.0}}, ...]
"""


def rerank(query: str, docs: list[Document], model, top_n: int) -> list[Document]:
    """Keep the top_n by score, stamping each kept doc's rerank_score onto its
    metadata (query-time only, never persisted)."""
    if not docs:
        return docs
    scores = model.predict([(query, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    kept = ranked[:top_n]
    for doc, score in kept:
        doc.metadata["rerank_score"] = float(score)
    return [doc for doc, _ in kept]


class LLMReranker:
    """Scores candidates in batched LLM calls; same ``.predict`` contract as a
    cross-encoder. All pairs in one call share the query."""

    def __init__(self, llm, snippet_chars: int = 500, batch_size: int = 40):
        self._llm = llm
        self._snippet_chars = snippet_chars
        self._batch_size = batch_size

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        question = pairs[0][0]
        texts = [p[1] for p in pairs]
        scores = [0.0] * len(texts)
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start:start + self._batch_size]
            for local_id, score in self._score_batch(question, batch).items():
                if 0 <= local_id < len(batch):
                    scores[start + local_id] = score
        return scores

    def _score_batch(self, question: str, texts: list[str]) -> dict[int, float]:
        lines = [f"{i}: {' '.join(text.split())[:self._snippet_chars]}"
                 for i, text in enumerate(texts)]
        raw = self._llm.invoke(
            _RERANK_PROMPT.format(question=question, excerpts="\n".join(lines))
        ).content
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return {}
        try:
            entries = json.loads(match.group(0))
        except (ValueError, TypeError):
            return {}
        out: dict[int, float] = {}
        for e in entries:
            try:
                out[int(e["id"])] = float(e["score"])
            except (KeyError, TypeError, ValueError):
                continue
        return out


def load_reranker(model_name: str):
    """Load the cross-encoder (downloads weights on first use)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def build_reranker(settings, llm=None):
    if settings.reranker_provider == "cross_encoder":
        return load_reranker(settings.reranker_model)
    if settings.reranker_provider == "llm":
        if llm is None:
            raise ValueError("reranker_provider='llm' requires an llm instance")
        return LLMReranker(
            llm,
            snippet_chars=settings.rerank_snippet_chars,
            batch_size=settings.rerank_batch_size,
        )
    raise ValueError(f"Unknown reranker_provider: {settings.reranker_provider}")
