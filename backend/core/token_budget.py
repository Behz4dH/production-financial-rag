"""Token counting and a per-request budget guard (course: cost_optimization.py)."""


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


def within_budget(text: str, max_tokens: int) -> tuple[bool, int]:
    n = count_tokens(text)
    return n <= max_tokens, n
