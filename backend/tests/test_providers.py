"""Provider factory: offline (providers are monkeypatched — no network/model)."""

import pytest
from langchain_core.embeddings import Embeddings

from core.config import Settings
from rag.providers import factory


def _settings(**over) -> Settings:
    base = {"groq_api_key": "test-key", "_env_file": None}
    base.update(over)
    return Settings(**base)


class _CountingEmbeddings(Embeddings):
    """Fake base embeddings that counts embed_documents calls."""

    def __init__(self):
        self.doc_calls = 0

    def embed_documents(self, texts):
        self.doc_calls += 1
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


def test_get_llm_uses_init_chat_model_with_settings(monkeypatch):
    captured = {}

    def fake_init(model, model_provider=None, api_key=None, temperature=None, max_retries=None, **kw):
        captured.update(model=model, provider=model_provider, api_key=api_key,
                        temperature=temperature, max_retries=max_retries)
        return "FAKE_LLM"

    monkeypatch.setattr(factory, "init_chat_model", fake_init)
    llm = factory.get_llm(_settings(primary_model="llama-3.1-8b-instant",
                                    llm_provider="groq", max_retries=5))
    assert llm == "FAKE_LLM"
    assert captured["model"] == "llama-3.1-8b-instant"
    assert captured["provider"] == "groq"
    assert captured["api_key"] == "test-key"
    assert captured["max_retries"] == 5


def test_get_embeddings_is_cache_backed_and_caches(monkeypatch, tmp_path):
    counting = _CountingEmbeddings()
    monkeypatch.setattr(factory, "_build_base_embeddings", lambda s: counting)

    s = _settings(embedding_cache_dir=str(tmp_path / "emb"),
                  embedding_model="fake-model")
    emb = factory.get_embeddings(s)

    assert isinstance(emb, Embeddings)
    # First call hits the underlying model; second identical call is served from cache.
    emb.embed_documents(["hello world"])
    emb.embed_documents(["hello world"])
    assert counting.doc_calls == 1
