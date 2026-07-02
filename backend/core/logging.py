"""Structured JSON logging for production log aggregation."""

import json
import logging
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        extra = getattr(record, "extra_data", None)
        if isinstance(extra, dict):
            log_obj.update(extra)
        return json.dumps(log_obj)


def get_logger(name: str = "financial_rag", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger
