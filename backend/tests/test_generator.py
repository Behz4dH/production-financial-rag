"""generate(): reasoning always populated, prompt selection by multi_source, refusal reason from the model."""

from langchain_core.documents import Document

from rag.generation.generator import generate, refusal
from rag.generation.schema import AnswerDraft


class _LLM:
    """Returns a fixed draft, and records the prompt it was invoked with so
    tests can check which instructions (base vs comparison) were used."""
    def __init__(self, answer="88.1", refused=False, reasoning="matched net income on p1."):
        self._draft = AnswerDraft(reasoning=reasoning, answer=answer, refused=refused)
        self.last_prompt = None

    def with_structured_output(self, schema):
        assert schema is AnswerDraft  # one unified schema now, always
        draft = self._draft

        class _S:
            def invoke(_self, prompt):
                self.last_prompt = prompt
                return draft
        return _S()


def _docs():
    return [Document(page_content="net income was 88.1", metadata={"source": "a.pdf", "page": 1})]


def test_generate_returns_the_models_reasoning():
    llm = _LLM(reasoning="excerpt [1] reports net income of 88.1 for the asked period.")
    result = generate("net income?", _docs(), llm)
    assert result.answer == "88.1"
    assert result.reasoning == "excerpt [1] reports net income of 88.1 for the asked period."


def test_generate_uses_comparison_instructions_for_multi_source():
    llm = _LLM()
    generate("who has more equity, A or B?", _docs(), llm, multi_source=True)
    assert "different currencies" in llm.last_prompt.to_string()


def test_generate_uses_base_instructions_for_single_source():
    llm = _LLM()
    generate("net income?", _docs(), llm, multi_source=False)
    assert "different currencies" not in llm.last_prompt.to_string()


def test_generation_refusal_reason_is_the_models_own_reasoning():
    # this is what replaced the old fixed "excerpts retrieved but
    # insufficient to answer" string -- the model has to say what's
    # actually missing or mismatched.
    llm = _LLM(answer="N/A", refused=True, reasoning="no excerpt covers the asked FY2023 period.")
    result = generate("net income in 2023?", _docs(), llm)
    assert result.refused is True
    assert result.reason == "no excerpt covers the asked FY2023 period."
    assert result.reasoning == result.reason


def test_refusal_has_no_reasoning():
    result = refusal("no filing matches X")
    assert result.refused is True
    assert result.reason == "no filing matches X"
    assert result.reasoning == ""


def test_generate_recovers_from_groq_tool_use_failed():
    """Groq's tool parser sometimes rejects the 8b model's '<function=...>'
    wrapper even when the JSON payload is valid — recover the draft from the
    error body instead of failing the whole request."""
    from langchain_core.documents import Document

    from rag.generation.generator import generate

    failed = ('<function=AnswerDraft> {"reasoning": "Excerpt [1] states it.", '
              '"answer": "88.1", "refused": false, "confidence": "high"}')

    class _ToolUseFailed(Exception):
        def __init__(self):
            super().__init__("Error code: 400 - tool_use_failed")
            self.body = {"error": {"code": "tool_use_failed",
                                   "failed_generation": failed}}

    class _LLM:
        def with_structured_output(self, schema):
            class _S:
                def invoke(self, _p):
                    raise _ToolUseFailed()
            return _S()

    docs = [Document(page_content="Profit for the Year 88.1",
                     metadata={"source": "a.pdf", "page": 1})]
    result = generate("net income?", docs, _LLM())
    assert result.answer == "88.1"
    assert result.refused is False


def test_generate_reraises_other_llm_errors():
    import pytest
    from langchain_core.documents import Document

    from rag.generation.generator import generate

    class _LLM:
        def with_structured_output(self, schema):
            class _S:
                def invoke(self, _p):
                    raise RuntimeError("rate limited")
            return _S()

    docs = [Document(page_content="x", metadata={"source": "a.pdf", "page": 1})]
    with pytest.raises(RuntimeError):
        generate("q?", docs, _LLM())


def test_generate_retries_once_when_tool_failure_payload_is_garbage():
    """If the tool_use_failed body can't be parsed, retry the call once
    (Groq generations vary run-to-run) instead of failing the request."""
    from langchain_core.documents import Document

    from rag.generation.generator import generate

    class _ToolUseFailedGarbage(Exception):
        def __init__(self):
            super().__init__("400 tool_use_failed")
            self.body = {"error": {"code": "tool_use_failed",
                                   "failed_generation": "<function=AnswerDraft> {truncated garba"}}

    calls = {"n": 0}

    class _LLM:
        def with_structured_output(self, schema):
            class _S:
                def invoke(self, _p):
                    calls["n"] += 1
                    if calls["n"] == 1:
                        raise _ToolUseFailedGarbage()
                    return schema(reasoning="Excerpt [1] states it.", answer="88.1",
                                  refused=False, confidence="high")
            return _S()

    docs = [Document(page_content="x", metadata={"source": "a.pdf", "page": 1})]
    result = generate("q?", docs, _LLM())
    assert result.answer == "88.1"
    assert calls["n"] == 2


def test_generate_gives_up_after_retry_also_fails_unparseably():
    import pytest
    from langchain_core.documents import Document

    from rag.generation.generator import generate

    class _AlwaysGarbage(Exception):
        def __init__(self):
            super().__init__("400 tool_use_failed")
            self.body = {"error": {"code": "tool_use_failed", "failed_generation": "not json"}}

    class _LLM:
        def with_structured_output(self, schema):
            class _S:
                def invoke(self, _p):
                    raise _AlwaysGarbage()
            return _S()

    docs = [Document(page_content="x", metadata={"source": "a.pdf", "page": 1})]
    with pytest.raises(Exception, match="tool_use_failed"):
        generate("q?", docs, _LLM())


def test_answer_draft_coerces_string_booleans():
    """Scout emits refused as the string "false"/"true"; Groq rejects that
    against the schema, but OUR recovery parse must accept and coerce it."""
    from rag.generation.schema import AnswerDraft

    d = AnswerDraft(reasoning="r", answer="a", refused="false", confidence="high")
    assert d.refused is False
    d2 = AnswerDraft(reasoning="r", answer="a", refused="true")
    assert d2.refused is True


def test_generate_recovers_scout_wrapped_tool_payload():
    """Scout's failed_generation is a JSON ARRAY of {name, parameters} — the
    draft lives under parameters, with string booleans."""
    from langchain_core.documents import Document

    from rag.generation.generator import generate

    failed = ('[\n  {\n    "name": "AnswerDraft",\n    "parameters": {\n'
              '      "reasoning": "Stated directly.",\n      "answer": "$24,393,946",\n'
              '      "confidence": "high",\n      "refused": "false"\n    }\n  }\n]')

    class _ToolUseFailed(Exception):
        def __init__(self):
            super().__init__("400 tool_use_failed")
            self.body = {"error": {"code": "tool_use_failed",
                                   "failed_generation": failed}}

    class _LLM:
        def with_structured_output(self, schema):
            class _S:
                def invoke(self, _p):
                    raise _ToolUseFailed()
            return _S()

    docs = [Document(page_content="Revenue $24,393,946", metadata={"source": "a.pdf", "page": 1})]
    result = generate("revenue?", docs, _LLM())
    assert result.answer == "$24,393,946"
    assert result.refused is False
