"""Token counting + per-request budget check."""

from core.token_budget import count_tokens


def test_count_tokens_positive():
    assert count_tokens("hello world foo bar") > 0


def test_count_tokens_scales_with_length():
    assert count_tokens("word " * 500) > count_tokens("short")


def test_count_with_margin_inflates_the_estimate():
    """cl100k undercounts Llama tokenization on dense numeric text (measured:
    '4,500 counted' prompts drew 413s at Groq's 6,000 cap). Budget checks must
    treat the count as an underestimate."""
    from core.token_budget import count_tokens, count_tokens_with_margin

    text = "Revenue 585.2 (391.5) 146,469 1,274,570 " * 50
    raw = count_tokens(text)
    assert count_tokens_with_margin(text) == int(raw * 1.25)
    assert count_tokens_with_margin(text) > raw
