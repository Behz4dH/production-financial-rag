"""Resolve a question's company to source filing(s).

The refusal lever here is company existence only: if no filing matches the
asked company, there is nothing to retrieve from and the answer is N/A --
decided before any retrieval or generation. Fiscal year is deliberately NOT
gated here -- a 10-K reports multi-year comparatives (a FY2022 filing's
balance sheet routinely includes FY2021 figures too), so a strict year match
at this stage produced false refusals on legitimately answerable questions.
Whether the retrieved text actually covers the asked year is left to
generation's own reasoning, which can see the real page content.
"""

import re
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag.providers.structured import invoke_structured

_PARSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Extract the company name(s) and the fiscal year the question asks about. "
         "Companies are usually named explicitly (often in quotes). If no year is "
         "stated, leave fiscal_year null. Return every company mentioned. "
         "If the question asks about the document collection itself — which "
         "companies, filings, or years are available — set "
         "asks_corpus_inventory=true and leave companies empty."),
        ("human", "{question}"),
    ]
)


class QueryEntities(BaseModel):
    companies: list[str] = Field(default_factory=list,
                                 description="Company name(s) the question is about")
    fiscal_year: str | None = Field(default=None, description="Asked fiscal year, e.g. '2022'")
    asks_corpus_inventory: bool = Field(
        default=False,
        description="True if the question is about the corpus itself (which "
                    "companies/filings/years are available), not about facts in a filing")


def parse_query(question: str, llm) -> QueryEntities:
    return invoke_structured(llm, QueryEntities, _PARSE_PROMPT.invoke({"question": question}))


_SUFFIXES = {"inc", "incorporated", "ltd", "limited", "plc", "corp", "corporation",
             "sa", "ag", "llc", "co", "company", "holdings", "group", "the"}


def _normalize(name: str) -> frozenset[str]:
    """Lowercased significant tokens, dropping punctuation and common suffixes."""
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    return frozenset(t for t in tokens if t not in _SUFFIXES)


def _matches(asked: str, candidate: str) -> bool:
    a, c = _normalize(asked), _normalize(candidate)
    if not a or not c:
        return False
    # match when one name's significant tokens are a subset of the other's
    return a <= c or c <= a


def build_index(doc_metadata: dict) -> list[dict]:
    records: list[dict] = []
    for source, meta in doc_metadata.items():
        names = [meta.get("company_name", "")] + list(meta.get("aliases") or [])
        for name in names:
            if name:
                records.append({"source": source, "name": name,
                                "fiscal_year": meta.get("fiscal_year")})
    return records


@dataclass
class Resolution:
    sources: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    @property
    def refuse(self) -> bool:
        return not self.sources


def resolve(entities: QueryEntities, index: list[dict]) -> Resolution:
    res = Resolution()
    for company in entities.companies:
        hit = next((rec["source"] for rec in index if _matches(company, rec["name"])), None)
        if hit is not None:
            if hit not in res.sources:
                res.sources.append(hit)
        else:
            res.unresolved.append(company)
    return res
