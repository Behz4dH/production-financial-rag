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
    assert s.enable_llm_guard is False
    assert s.rate_limit == "20/minute"


def test_is_production_property():
    assert _make(app_env="development").is_production is False
    assert _make(app_env="production").is_production is True


def test_env_overrides_are_typed(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("ENABLE_LLM_GUARD", "true")
    s = Settings(_env_file=None)
    assert s.groq_api_key == "env-key"
    assert s.top_k == 7 and isinstance(s.top_k, int)
    assert s.enable_llm_guard is True


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
