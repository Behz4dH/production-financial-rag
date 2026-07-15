"""Tests for API request/response models."""

import pytest
from pydantic import ValidationError

from app.models import ChatRequest, ChatResponse, Citation


def test_chat_request_defaults():
    req = ChatRequest(message="What was TransUnion's total assets in 2023?")
    assert req.mode is None


def test_chat_request_rejects_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_chat_request_rejects_too_long_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="x" * 2001)


def test_chat_response_auto_timestamp_and_defaults():
    resp = ChatResponse(
        response="N/A",
        model_used="llama-3.1-8b-instant",
        mode="hybrid",
        processing_time_ms=12.3,
    )
    assert resp.refused is False
    assert resp.cached is False
    assert resp.citations == []
    assert "T" in resp.timestamp  # ISO-8601


def test_citation_shape():
    c = Citation(company="TransUnion", fiscal_year="2022", page=115, source="transunion.pdf")
    assert c.page == 115
