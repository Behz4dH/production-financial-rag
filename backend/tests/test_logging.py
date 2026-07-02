"""Tests for structured JSON logging."""

import json
import logging

from core.logging import JSONFormatter, get_logger


def test_formatter_emits_valid_json_with_core_fields():
    record = logging.LogRecord(
        name="x", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello", args=(), exc_info=None, func="myfunc",
    )
    payload = json.loads(JSONFormatter().format(record))
    assert payload["level"] == "INFO"
    assert payload["message"] == "hello"
    assert payload["function"] == "myfunc"
    assert "timestamp" in payload


def test_formatter_merges_extra_data():
    record = logging.LogRecord(
        name="x", level=logging.INFO, pathname=__file__, lineno=1,
        msg="req", args=(), exc_info=None,
    )
    record.extra_data = {"latency_ms": 12.5, "mode": "hybrid"}
    payload = json.loads(JSONFormatter().format(record))
    assert payload["latency_ms"] == 12.5
    assert payload["mode"] == "hybrid"


def test_get_logger_does_not_duplicate_handlers():
    a = get_logger("dup_test")
    b = get_logger("dup_test")
    assert a is b
    assert len(a.handlers) == 1


def test_get_logger_respects_level():
    logger = get_logger("level_test", level="WARNING")
    assert logger.level == logging.WARNING
