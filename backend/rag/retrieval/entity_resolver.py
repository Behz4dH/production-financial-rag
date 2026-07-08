"""Resolve a question's company + fiscal year to source filing(s).

The refusal lever: if no filing matches the asked company (or the asked year
differs from the filing's fiscal_year), there is nothing to retrieve from and
the answer is N/A — decided here, before any retrieval or generation.
"""

import re
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

_PARSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Extract the company name(s) and the fiscal year the question asks about. "
         "Companies are usually named explicitly (often in quotes). If no year is "
         "stated, leave fiscal_year null. Return every company mentioned."),
        ("human", "{question}"),
    ]
)


class QueryEntities(BaseModel):
    companies: list[str] = Field(default_factory=list,
                                 description="Company name(s) the question is about")
    fiscal_year: str | None = Field(default=None, description="Asked fiscal year, e.g. '2022'")


def parse_query(question: str, llm) -> QueryEntities:
    return llm.with_structured_output(QueryEntities).invoke(_PARSE_PROMPT.invoke({"question": question}))


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
    """One record per (source, matchable name) — the canonical company_name is
    carried on every record (not just the one where name == company_name) so
    a match via an alias still tells the caller the filing's real name."""
    records: list[dict] = []
    for source, meta in doc_metadata.items():
        company_name = meta.get("company_name", "")
        names = [company_name] + list(meta.get("aliases") or [])
        for name in names:
            if name:
                records.append({"source": source, "name": name,
                                "company_name": company_name,
                                "fiscal_year": meta.get("fiscal_year")})
    return records


@dataclass
class ResolvedEntity:
    source: str
    company_name: str
    fiscal_year: str | None


@dataclass
class Resolution:
    sources: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    entities: list[ResolvedEntity] = field(default_factory=list)

    @property
    def refuse(self) -> bool:
        return not self.sources


def resolve(entities: QueryEntities, index: list[dict]) -> Resolution:
    res = Resolution()
    for company in entities.companies:
        hit = None
        for rec in index:
            if _matches(company, rec["name"]):
                year_ok = entities.fiscal_year is None or rec["fiscal_year"] == entities.fiscal_year
                if year_ok:
                    hit = rec
                    break
        if hit is not None:
            if hit["source"] not in res.sources:
                res.sources.append(hit["source"])
                res.entities.append(ResolvedEntity(
                    source=hit["source"], company_name=hit["company_name"],
                    fiscal_year=hit["fiscal_year"]))
        else:
            res.unresolved.append(company)
    return res
