"""Robust structured-output invocation shared by ALL LLM structured calls.

Groq-hosted models (Scout especially) emit tool payloads that fail Groq's
strict schema validation over pure formatting: string booleans ("true"),
stringified arrays ("[]"), envelope wrapping ([{"name","parameters"}]). The
correct values are recoverable from the error body — every structured call
site (entity parse, generation, metadata extraction) must share one recovery
path instead of each rediscovering this failure class.
"""

import pytest
from pydantic import BaseModel, Field

from rag.providers.structured import invoke_structured


class _Entities(BaseModel):
    companies: list[str] = Field(default_factory=list)
    fiscal_year: str | None = None
    asks_corpus_inventory: bool = False


def _tool_failed_exc(failed_generation: str) -> Exception:
    exc = Exception("400 tool_use_failed")
    exc.body = {"error": {"code": "tool_use_failed",
                          "failed_generation": failed_generation}}
    return exc


class _RaisingLLM:
    def __init__(self, exc):
        self._exc = exc
        self.calls = 0

    def with_structured_output(self, schema):
        outer = self

        class _S:
            def invoke(self, _p):
                outer.calls += 1
                raise outer._exc
        return _S()


def test_recovers_stringified_arrays_booleans_and_nulls():
    failed = ('[{"name": "QueryEntities", "parameters": {'
              '"asks_corpus_inventory": "true", "companies": "[]", "fiscal_year": "null"}}]')
    llm = _RaisingLLM(_tool_failed_exc(failed))
    out = invoke_structured(llm, _Entities, "prompt")
    assert out.asks_corpus_inventory is True
    assert out.companies == []
    assert out.fiscal_year is None
    assert llm.calls == 1  # recovered from the first failure, no retry needed


def test_string_values_that_are_not_json_stay_strings():
    failed = ('[{"name": "QueryEntities", "parameters": {'
              '"companies": "[\\"Holley Inc.\\"]", "fiscal_year": "2022"}}]')
    out = invoke_structured(_RaisingLLM(_tool_failed_exc(failed)), _Entities, "p")
    assert out.companies == ["Holley Inc."]
    assert out.fiscal_year == "2022"  # a plain string field is left alone


def test_non_tool_failures_reraise():
    llm = _RaisingLLM(RuntimeError("rate limited"))
    with pytest.raises(RuntimeError):
        invoke_structured(llm, _Entities, "p")
    assert llm.calls == 1
