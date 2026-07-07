"""LangSmith tracing setup. Opt-in: sets the env vars LangChain reads so tracing
is automatic, or no-ops when disabled/unset (course: langsmith_setup.py)."""

import os

from core.config import Settings


def configure_tracing(settings: Settings) -> bool:
    if not settings.langsmith_tracing or not settings.langsmith_api_key:
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    return True
