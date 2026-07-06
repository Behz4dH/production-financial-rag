"""Refusal-first grounded generation via structured output."""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from rag.generation.schema import RAGAnswer

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You answer questions about company financial filings using ONLY the "
         "provided excerpts. If the excerpts do not contain the answer, set "
         "refused=true and answer exactly 'N/A'. Never invent figures. Cite the "
         "source and page for every figure you report."),
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


def generate(question: str, docs: list[Document], llm) -> RAGAnswer:
    structured = llm.with_structured_output(RAGAnswer)
    prompt = _PROMPT.invoke({"question": question, "context": format_context(docs)})
    return structured.invoke(prompt)


def refusal(reason: str) -> RAGAnswer:
    return RAGAnswer(answer="N/A", citations=[], refused=True, confidence="high")
