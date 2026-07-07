"""Token counting + per-request budget check."""

from core.token_budget import count_tokens, within_budget


def test_count_tokens_positive():
    assert count_tokens("hello world foo bar") > 0


def test_within_budget_true_and_false():
    ok, n = within_budget("short question", max_tokens=1000)
    assert ok is True and n > 0
    ok2, n2 = within_budget("word " * 5000, max_tokens=100)
    assert ok2 is False and n2 > 100
