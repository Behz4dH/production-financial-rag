"""Extract reporting-entity metadata from a filing's first pages via the LLM,
cached to a JSON file so re-ingestion is deterministic and reviewable."""

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured metadata from the opening pages of a company "
            "annual report or SEC filing. Identify the SINGLE reporting entity (the "
            "company the report is FOR, not its auditor, exchange, or subsidiaries). "
            "Use the fiscal year the financial statements cover. If a field is not "
            "stated, leave it null. Currency is the reporting currency of the primary "
            "financial statements (ISO code like USD, EUR, GBP, JPY, CHF)."
            "be careful about rotated or mirrored text like report coming off as troper.",
        ),
        ("human", "Opening pages:\n\n{text}"),
    ]
)


class DocumentMetadata(BaseModel):
    company_name: str = Field(description="Reporting entity's name")
    # Nullable in the schema because small models emit `null` (not `[]`) for an
    # empty list, and strict tool-schema validation (e.g. Groq) rejects null
    # against a non-nullable array before Pydantic ever runs. Coerced to [] below.
    aliases: list[str] | None = Field(default=None,
                                      description="Other names/legal forms for the same entity")
    ticker: str | None = Field(default=None, description="Stock ticker if stated")
    fiscal_year: str = Field(description="Fiscal year the statements cover, e.g. '2022'")
    period_end_date: str | None = Field(default=None,
                                        description="Fiscal period end, ISO date if known")
    reporting_currency: str | None = Field(default=None, description="ISO currency code")
    report_type: str | None = Field(default=None, description="e.g. '10-K', 'Annual Report'")

    @field_validator("aliases", mode="before")
    @classmethod
    def _null_aliases_to_empty(cls, v: object) -> list:
        return v or []


def first_pages_text(pages: list[Document], n: int, max_chars: int | None = None) -> str:
    """Concatenate the first ``n`` pages, optionally truncated to ``max_chars``.

    The truncation bounds the tokens sent to the LLM for metadata extraction —
    dense filings can otherwise exceed provider per-request/rate limits, and the
    reporting entity / fiscal year / currency all appear on the opening pages.
    """
    text = "\n\n".join(p.page_content for p in pages[:n])
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars]
    return text


def extract_metadata(doc_text: str, llm) -> DocumentMetadata:
    structured = llm.with_structured_output(DocumentMetadata)
    prompt = _PROMPT.invoke({"text": doc_text})
    return structured.invoke(prompt)


def _read_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return {}


def load_or_extract(
    source_name: str,
    pages: list[Document],
    llm,
    cache_path: str,
    n_pages: int,
    max_chars: int | None = None,
) -> DocumentMetadata:
    path = Path(cache_path)
    cache = _read_cache(path)
    if source_name in cache:
        return DocumentMetadata(**cache[source_name])

    md = extract_metadata(first_pages_text(pages, n_pages, max_chars), llm)
    cache[source_name] = md.model_dump()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return md
