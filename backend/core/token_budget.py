"""Token counting and a per-request budget guard (course: cost_optimization.py)."""

# count_tokens uses the cl100k tokenizer, which undercounts Llama-family
# tokenization on dense numeric/financial text — measured in practice as
# "4,500 counted" prompts drawing 413s at Groq's 6,000-token cap. Budget
# comparisons must treat the count as an underestimate.
TOKEN_MARGIN = 1.25


def count_tokens_with_margin(text: str, model: str = "gpt-4o") -> int:
    """Conservative token estimate for budget enforcement."""
    return int(count_tokens(text, model) * TOKEN_MARGIN)


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback estimate if tiktoken/model data is unavailable.
        return max(1, int(len(text.split()) * 1.3))
