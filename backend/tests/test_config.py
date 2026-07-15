"""Tests for centralized settings."""

import pytest

from core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _groq_api_key_env(request, monkeypatch):
    """Isolate GROQ_API_KEY handling for this module only.

    test_required_key_missing_raises needs NO key present so that
    Settings(_env_file=None) actually raises. Every other test in this
    module calls bare Settings()/get_settings() and needs a dummy key so
    it doesn't depend on a real .env file or network access. Scoped to
    this module (not conftest.py) so it can't leak into unrelated tests
    elsewhere in the suite.
    """
    if request.node.name == "test_required_key_missing_raises":
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GROQ_API_KEY", "test-api-key")


def _make(**overrides) -> Settings:
    """Build Settings without reading a .env file, with a required key default."""
    base = {"groq_api_key": "test-key", "_env_file": None}
    base.update(overrides)
    return Settings(**base)


def test_required_key_missing_raises():
    with pytest.raises(Exception):
        Settings(_env_file=None)  # no groq_api_key provided anywhere


def test_defaults_are_applied():
    s = _make()
    assert s.primary_model == "llama-3.1-8b-instant"
    assert s.embedding_model == "BAAI/bge-small-en-v1.5"
    assert s.vector_store == "chroma"
    assert s.top_k == 20
    assert s.top_n == 5
    assert s.rate_limit == "20/minute"
    assert s.retry_base_delay == 0.5
    # Reranking is where an absolute score gates refusal — the 8b model
    # degenerates to all-zero scores on batches of weak candidates, so the
    # scoring model defaults to the stronger 70b.
    assert s.reranker_llm_model == "llama-3.3-70b-versatile"
    # Question+context budget; system prompt + tool schema add ~1000 tokens,
    # and the whole request must stay under Groq free-tier 6000 TPM for the
    # generation model — an 8000-token prompt can NEVER succeed there.
    assert s.max_context_tokens == 4500


def test_is_production_property():
    assert _make(app_env="development").is_production is False
    assert _make(app_env="production").is_production is True


def test_env_overrides_are_typed(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    monkeypatch.setenv("TOP_K", "7")
    s = Settings(_env_file=None)
    assert s.groq_api_key == "env-key"
    assert s.top_k == 7 and isinstance(s.top_k, int)


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
