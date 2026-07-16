"""Runner scores predictions with an injected fake answer_fn (no LLM)."""

from eval.golden import GoldenItem
from eval.runner import run_mode
from rag.generation.schema import RAGAnswer


def _items():
    return [
        GoldenItem(question="petra net income 2022?", answer_type="number",
                   answers=[88100000], category="retrieval"),
        GoldenItem(question="crossfirst liabilities 2023?", answer_type="number",
                   answers=["N/A"], category="hallucination"),
    ]


def test_run_mode_scores_each_item():
    def fake_answer(question, mode, deps):
        if "petra" in question:
            return RAGAnswer(answer="88.1 million", refused=False)
        return RAGAnswer(answer="N/A", refused=True)

    rows = run_mode(_items(), "hybrid", deps=None, answer_fn=fake_answer)
    assert len(rows) == 2
    assert rows[0].correct is True   # "88.1 million" ≈ 88,100,000
    assert rows[1].correct is True   # refusal matches N/A


def test_run_mode_captures_errors():
    def boom(question, mode, deps):
        raise RuntimeError("provider down")

    rows = run_mode(_items()[:1], "hybrid", deps=None, answer_fn=boom)
    assert rows[0].correct is False
    assert "provider down" in rows[0].error


class _RateLimited(Exception):
    def __init__(self):
        super().__init__("429 too many requests")
        self.status_code = 429


def test_run_mode_retries_rate_limits_until_success():
    """Free-tier TPM limits make sustained eval runs 429 — the runner must
    self-pace with retries instead of recording dead rows."""
    calls = {"n": 0}

    def flaky_answer(question, mode, deps):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _RateLimited()
        return RAGAnswer(answer="N/A", refused=True)

    items = [GoldenItem(question="q?", answer_type="number", answers=["N/A"],
                        category="hallucination")]
    rows = run_mode(items, "basic", deps=None, answer_fn=flaky_answer,
                    retry_base_delay=0.0)
    assert rows[0].error == ""
    assert rows[0].correct is True
    assert calls["n"] == 3


def test_run_mode_does_not_retry_deterministic_errors():
    calls = {"n": 0}

    class _BadRequest(Exception):
        status_code = 400

    def bad_answer(question, mode, deps):
        calls["n"] += 1
        raise _BadRequest("bad request")

    items = [GoldenItem(question="q?", answer_type="number", answers=["1"])]
    rows = run_mode(items, "basic", deps=None, answer_fn=bad_answer,
                    retry_base_delay=0.0)
    assert calls["n"] == 1  # no pointless retries against a 400
    assert rows[0].error != ""


def test_run_mode_scores_alias_answers_via_deps_entity_index():
    class _Deps:
        entity_index = [
            {"source": "mol.pdf", "name": "Mitsui O.S.K. Lines, Ltd.", "fiscal_year": "2022"},
            {"source": "mol.pdf", "name": "MOL", "fiscal_year": "2022"},
        ]

    def alias_answer(question, mode, deps):
        return RAGAnswer(answer="MOL Group", refused=False)

    items = [GoldenItem(question="who had higher equity?", answer_type="name",
                        answers=["MITSUI O.S.K. LINES"])]
    rows = run_mode(items, "hybrid", deps=_Deps(), answer_fn=alias_answer,
                    retry_base_delay=0.0)
    assert rows[0].correct is True
