"""Structured-output invocation with recovery from Groq's strict validation.

Models emit substantively-correct payloads with formal defects — string
booleans, stringified arrays, ``[{"name", "parameters"}]`` envelopes,
``<function=...>`` prefixes — which Groq rejects as ``tool_use_failed``
while returning the raw payload in the error body. Every structured call
site shares this path: parse the rejected payload leniently; failing that,
retry the call once; only then raise.
"""

import json
import re


def invoke_structured(llm, schema, prompt_value):
    """`llm.with_structured_output(schema).invoke(prompt_value)` with recovery."""
    structured = llm.with_structured_output(schema)
    try:
        return structured.invoke(prompt_value)
    except Exception as exc:  # noqa: BLE001 — recover tool_use_failed, re-raise the rest
        recovered = _recover(exc, schema)
        if recovered is not None:
            return recovered
        if not _is_tool_use_failed(exc):
            raise
        try:
            return structured.invoke(prompt_value)
        except Exception as exc2:  # noqa: BLE001
            recovered = _recover(exc2, schema)
            if recovered is None:
                raise
            return recovered


def _is_tool_use_failed(exc: Exception) -> bool:
    body = getattr(exc, "body", None)
    return isinstance(body, dict) and body.get("error", {}).get("code") == "tool_use_failed"


def _recover(exc: Exception, schema):
    if not _is_tool_use_failed(exc):
        return None
    failed = getattr(exc, "body", {}).get("error", {}).get("failed_generation", "")
    for candidate in _payload_candidates(failed):
        for attempt in (candidate, _json_normalized(candidate)):
            try:
                return schema.model_validate(attempt)
            except ValueError:
                continue
    return None


def _payload_candidates(failed: str):
    """Possible schema dicts inside a failed_generation string."""
    try:
        parsed = json.loads(failed)
    except ValueError:
        match = re.search(r"\{.*\}", failed, re.DOTALL)
        if not match:
            return
        try:
            parsed = json.loads(match.group(0))
        except ValueError:
            return
    items = parsed if isinstance(parsed, list) else [parsed]
    for item in items:
        if not isinstance(item, dict):
            continue
        yield item.get("parameters") if isinstance(item.get("parameters"), dict) else item


def _json_normalized(candidate: dict) -> dict:
    """Decode stringified JSON values ("[]", "true", "null", '["A"]') while
    leaving genuine strings alone — numbers are deliberately NOT decoded so a
    string answer like "88.1" stays a string."""
    out = {}
    for key, value in candidate.items():
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, (list, dict, bool)) or parsed is None:
                    value = parsed
            except ValueError:
                pass
        out[key] = value
    return out
