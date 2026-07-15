"""Robust structured-output invocation for Groq-hosted models.

Groq validates tool arguments STRICTLY against the schema before we ever see
them, and models — Scout especially — routinely emit payloads that are
substantively correct but formally off: booleans as strings ("true"), arrays
as strings ("[]"), the whole call wrapped in `[{"name", "parameters"}]`, or
the 8b's `<function=...> {...}` prefix. Groq rejects these with a
`tool_use_failed` 400 whose body carries the raw payload verbatim.

Discarding a correct answer over a formatting quibble is strictly worse than
recovering it, so every structured call site (entity parse, generation,
metadata extraction) goes through here: try the call; on tool_use_failed,
parse the failed payload leniently; if even that fails, retry the call once
(Groq generations vary run-to-run); only then raise.
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
