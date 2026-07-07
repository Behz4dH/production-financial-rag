"""Token counting + per-request budget check."""

from core.token_budget import count_tokens


def test_count_tokens_positive():
    assert count_tokens("hello world foo bar") > 0


def test_count_tokens_scales_with_length():
    assert count_tokens("word " * 500) > count_tokens("short")
