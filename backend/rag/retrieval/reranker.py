"""Final-stage reranking: score each (query, chunk) pair and keep the best.

Two interchangeable rerankers, selected by ``settings.reranker_provider``:

- ``cross_encoder`` — a local ``sentence_transformers`` cross-encoder. Fast and
  offline, but it systematically scores fluent prose above terse tabular text,
  so it buries the dense financial-table rows this corpus is full of.
- ``llm`` (default) — one LLM call that scores every candidate for whether it
  actually contains the asked-for figure. Recognizes table rows the
  cross-encoder misses, and its score doubles as the refusal signal
  (``relevance_refusal_reason`` in ``rag.query``).

Both implement the same duck-typed contract — ``.predict(pairs) -> list[float]``
— so ``rerank`` and the whole query path are agnostic to which one is loaded.
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
    """Rank by reranker score, keep the top_n, and stamp each kept doc's score
    onto its metadata (query-time only — never persisted) so callers can judge
    how relevant the best match actually was, not just whether any match
    exists."""
    if not docs:
        return docs
    scores = model.predict([(query, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    kept = ranked[:top_n]
    for doc, score in kept:
        doc.metadata["rerank_score"] = float(score)
    return [doc for doc, _ in kept]


class LLMReranker:
    """Scores candidates with a single LLM call, exposing the same
    ``.predict(pairs) -> list[float]`` contract as a cross-encoder.

    All pairs in one ``rerank`` call share the query (``pairs[i][0]``); the
    candidate texts are ``pairs[i][1]``. Long candidates are truncated to
    ``snippet_chars`` — this must stay generous (default 500): a multi-year
    table row can push the asked-for period's value hundreds of characters in.
    """

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
        lines = []
        for i, text in enumerate(texts):
            snippet = " ".join(text.split())[:self._snippet_chars]
            lines.append(f"{i}: {snippet}")
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
    """Pick the reranker by ``settings.reranker_provider``.

    ``llm`` is required for the ``llm`` provider (the caller passes the same
    chat model used for generation).
    """
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
