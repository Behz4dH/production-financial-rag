"""Token counting and the margin used for budget enforcement."""

# cl100k undercounts Llama-family tokenization on numeric-dense text by up to
# ~25%; budget checks must treat the count as an underestimate.
TOKEN_MARGIN = 1.25


def count_tokens_with_margin(text: str, model: str = "gpt-4o") -> int:
    """Conservative estimate for comparisons against a hard provider cap."""
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
        return max(1, int(len(text.split()) * 1.3))
