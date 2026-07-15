"""Refusal-first grounded generation via structured output."""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from core.token_budget import count_tokens_with_margin
from rag.generation.schema import AnswerDraft, Citation, RAGAnswer
from rag.providers.structured import invoke_structured

_BASE_INSTRUCTIONS = (
    "You answer questions about company financial filings using ONLY the "
    "provided excerpts.\n\n"
    "Before answering, work through this procedure and show it in "
    "`reasoning`:\n"
    "1. Define precisely what the question's metric, entity, and period "
    "actually mean -- what is being measured, for whom, for when. Do this "
    "before looking for it in the excerpts, not after.\n"
    "2. Examine each excerpt against that definition: does it state that "
    "exact thing, or something merely similar, broader, narrower, or "
    "related?\n"
    "3. Accept ONLY an excerpt whose meaning exactly matches. A related-but-"
    "different line item, an aggregated total that includes the target, or "
    "a component of the target are near-misses, not matches, and must be "
    "rejected -- even if a real number is sitting right there.\n"
    "4. Never derive, calculate, or estimate the answer from other figures "
    "in the excerpts, no matter how straightforward it looks. If the metric "
    "itself isn't directly stated, that's N/A -- not a calculation to run.\n"
    "5. If any doubt remains about whether an excerpt truly matches, "
    "default to refusing. Guessing is worse than refusing.\n\n"
    "Filings often report multiple fiscal years side by side in the same "
    "table (e.g. a FY2022 balance sheet showing FY2021 comparatives too) -- "
    "entity resolution does not check the asked year against the filing's "
    "year, so confirming the period is part of step 2, not assumed from the "
    "question alone. Filings also often report both a GAAP/statutory figure "
    "and a non-GAAP 'adjusted' or 'underlying' variant of the same metric "
    "(e.g. 'net profit' vs 'adjusted net profit'); use the GAAP/statutory "
    "(as-reported) figure unless the question explicitly asks for the "
    "adjusted or underlying measure.\n\n"
    "Example of a correct refusal:\n"
    "Question: \"What was the R&D equipment, at cost, for Acme Corp?\"\n"
    "Excerpts contain: \"Property and equipment, net: $12.5M\" (broader, and "
    "net of depreciation -- not \"at cost\") and \"Accumulated depreciation, "
    "machinery: $1.1M\" (a depreciation figure, not the equipment's cost). "
    "Neither states R&D equipment at cost. Correct move: both are near-"
    "misses, not matches -- refused=true, answer='N/A', reasoning names "
    "both excerpts considered and explains why each falls short.\n\n"
    "If no excerpt actually matches after this procedure, set refused=true, "
    "answer exactly 'N/A', and say in `reasoning` exactly what's missing or "
    "mismatched. Never invent figures."
)

_COMPARISON_INSTRUCTIONS = _BASE_INSTRUCTIONS + (
    " The compared companies may report in different currencies — if so, "
    "convert to a common currency using your general knowledge of "
    "approximate exchange rates before comparing, and show that work in "
    "`reasoning` before giving your final `answer`."
)

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _BASE_INSTRUCTIONS), ("human", "Question:\n{question}\n\nExcerpts:\n{context}")]
)

_COMPARISON_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _COMPARISON_INSTRUCTIONS), ("human", "Question:\n{question}\n\nExcerpts:\n{context}")]
)


def format_context(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        # Company name over the hash filename: compare questions require the
        # model to attribute each excerpt to a company.
        src = d.metadata.get("company") or d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        parts.append(f"[{i}] ({src} p{page}) {d.page_content}")
    return "\n\n".join(parts)


def _citations_from(docs: list[Document]) -> list[Citation]:
    """Derive citations from the retrieved chunks (deduped by source+page) —
    the LLM never produces these, so they always reflect real retrieval."""
    seen: set[tuple] = set()
    cites: list[Citation] = []
    for d in docs:
        source = d.metadata.get("source", "")
        page = int(d.metadata.get("page", 0) or 0)
        if (source, page) not in seen:
            seen.add((source, page))
            cites.append(Citation(source=source, page=page,
                                  company=d.metadata.get("company", ""),
                                  fiscal_year=d.metadata.get("fiscal_year", "")))
    return cites


def _fit_context(question: str, docs: list[Document], max_tokens: int,
                 multi_source: bool = False) -> list[Document]:
    """Drop lowest-ranked docs (from the end; they arrive best-first) until
    question + context fits the budget, using the margined token count.
    Always keeps at least one doc — and with multi_source, at least one doc
    per source, so a compare question can't lose a company entirely to
    trimming."""
    kept = list(docs)
    while len(kept) > 1 and count_tokens_with_margin(question + format_context(kept)) > max_tokens:
        for i in range(len(kept) - 1, -1, -1):
            doc_source = kept[i].metadata.get("source")
            is_last_of_source = sum(1 for d in kept
                                    if d.metadata.get("source") == doc_source) == 1
            if multi_source and is_last_of_source:
                continue
            kept.pop(i)
            break
        else:
            break  # every remaining doc is its source's last — stop trimming
    return kept


def generate(question: str, docs: list[Document], llm,
             max_tokens: int | None = None, multi_source: bool = False) -> RAGAnswer:
    """multi_source=True (more than one source resolved) adds a currency-
    conversion instruction to the prompt; the schema is unchanged."""
    if max_tokens is not None:
        docs = _fit_context(question, docs, max_tokens, multi_source=multi_source)
    prompt = _COMPARISON_PROMPT if multi_source else _PROMPT
    messages = prompt.invoke({"question": question, "context": format_context(docs)})
    draft = invoke_structured(llm, AnswerDraft, messages)
    return RAGAnswer(
        answer=draft.answer,
        refused=draft.refused,
        confidence=draft.confidence,
        citations=[] if draft.refused else _citations_from(docs),
        reason=draft.reasoning if draft.refused else "",
        reasoning=draft.reasoning,
    )


def refusal(reason: str) -> RAGAnswer:
    return RAGAnswer(answer="N/A", citations=[], refused=True, confidence="high", reason=reason)
