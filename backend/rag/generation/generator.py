"""Refusal-first grounded generation via structured output."""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from rag.generation.schema import AnswerDraft, Citation, RAGAnswer

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You answer questions about company financial filings using ONLY the "
         "provided excerpts. If the excerpts do not contain the answer, set "
         "refused=true and answer exactly 'N/A'. Never invent figures."),
        ("human", "Question:\n{question}\n\nExcerpts:\n{context}"),
    ]
)


def format_context(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
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
            cites.append(Citation(source=source, page=page))
    return cites


def generate(question: str, docs: list[Document], llm) -> RAGAnswer:
    draft = llm.with_structured_output(AnswerDraft).invoke(
        _PROMPT.invoke({"question": question, "context": format_context(docs)})
    )
    return RAGAnswer(
        answer=draft.answer,
        refused=draft.refused,
        confidence=draft.confidence,
        citations=[] if draft.refused else _citations_from(docs),
    )


def refusal(reason: str) -> RAGAnswer:
    return RAGAnswer(answer="N/A", citations=[], refused=True, confidence="high")
