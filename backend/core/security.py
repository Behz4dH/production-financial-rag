"""Request-path security: prompt-injection screening and input PII masking.

Masking is input-only: answers derive from public filings, and masking
generated output corrupts legitimate figures (a plain 10-digit share count
matches the phone pattern).
"""

import re

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"forget\s+(all\s+)?previous",
    r"new\s+instructions:",
    r"system\s*prompt",
    r"pretend\s+you\s+are",
    r"bypass\s+(all\s+)?restrictions",
]

_PII = {
    "email": (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL REDACTED]"),
    "phone": (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE REDACTED]"),
    "ssn": (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]"),
    "credit_card": (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CARD REDACTED]"),
    "ip": (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP REDACTED]"),
}


class InputSanitizer:
    def __init__(self):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

    def is_suspicious(self, text: str) -> tuple[bool, str | None]:
        for pat in self._patterns:
            if pat.search(text):
                return True, f"injection pattern: {pat.pattern}"
        return False, None

    def sanitize(self, text: str) -> str:
        text = re.sub(r"[-=]{3,}", "", text)
        return text.strip()


class PIIDetector:
    def mask(self, text: str) -> str:
        for pattern, replacement in _PII.values():
            text = re.sub(pattern, replacement, text)
        return text


_sanitizer = InputSanitizer()
_pii = PIIDetector()


def screen_input(text: str) -> tuple[bool, str]:
    """(blocked, cleaned). Blocked on injection; otherwise sanitized + masked."""
    suspicious, _ = _sanitizer.is_suspicious(text)
    if suspicious:
        return True, text
    return False, _pii.mask(_sanitizer.sanitize(text))
