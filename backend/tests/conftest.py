"""Pytest configuration and fixtures."""

import os
import pytest


@pytest.fixture(autouse=True)
def ensure_groq_key_for_most_tests(request, monkeypatch):
    """Set GROQ_API_KEY for all tests except test_required_key_missing_raises."""
    if request.node.name != "test_required_key_missing_raises":
        monkeypatch.setenv("GROQ_API_KEY", "test-api-key")
    # For test_required_key_missing_raises, ensure GROQ_API_KEY is not set
    else:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
