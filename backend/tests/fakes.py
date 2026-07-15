"""Shared test fakes. Plain import, not a conftest.py — the project avoids
repo-wide conftest.py (implicit auto-discovery magic); every test file
imports what it needs explicitly instead."""

from langchain_core.embeddings import Embeddings


class HashEmbeddings(Embeddings):
    """Deterministic 8-dim embedding from token hashes — no network, no model."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * 8
        for tok in text.lower().split():
            v[hash(tok) % 8] += 1.0
        return v

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
