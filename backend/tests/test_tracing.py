"""Tracing is opt-in: no key -> no-op; key -> env configured."""

import os

from core.config import Settings
from core.tracing import configure_tracing


def _settings(**over):
    base = {"groq_api_key": "k", "_env_file": None}
    base.update(over)
    return Settings(**base)


def test_tracing_noop_without_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert configure_tracing(_settings(langsmith_tracing=False)) is False


def test_tracing_configures_env_when_enabled(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    enabled = configure_tracing(_settings(
        langsmith_tracing=True, langsmith_api_key="ls-key", langsmith_project="proj"))
    assert enabled is True
    assert os.environ["LANGSMITH_API_KEY"] == "ls-key"
    assert os.environ["LANGSMITH_PROJECT"] == "proj"
