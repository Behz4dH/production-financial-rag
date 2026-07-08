"""Agentic query path as a LangGraph StateGraph (course: 04_agentic_rag.py).

resolve -> [refuse | retrieve] -> grade -> [generate | rewrite->retrieve | refuse]
The grade step checks relevance, not just presence: did retrieval return
anything, and does the top reranked chunk clear refusal_score_threshold?
If not and retries remain, retry retrieval; otherwise refuse.
"""

from typing import Optional
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from rag.generation.generator import generate, refusal
from rag.generation.schema import RAGAnswer
from rag.query import QueryDeps, _retrieve, relevance_refusal_reason
from rag.retrieval.entity_resolver import parse_query, resolve
from rag.trace import TraceRecorder


class AgentState(TypedDict):
    question: str
    sources: list[str]
    unresolved: list[str]
    docs: list
    answer: Optional[RAGAnswer]
    retries: int


def build_agentic_app(deps: QueryDeps, trace: TraceRecorder | None = None):
    def resolve_node(state: AgentState) -> dict:
        entities = parse_query(state["question"], deps.llm)
        if trace is not None:
            trace.record("parse_query", companies=entities.companies, fiscal_year=entities.fiscal_year)
        res = resolve(entities, deps.entity_index)
        if trace is not None:
            trace.record("resolve_entities", sources=res.sources, unresolved=res.unresolved)
        return {"sources": res.sources, "unresolved": res.unresolved}

    def retrieve_node(state: AgentState) -> dict:
        docs = _retrieve(state["question"], "hybrid", state["sources"], deps, trace=trace)
        return {"docs": docs}

    def generate_node(state: AgentState) -> dict:
        result = generate(state["question"], state["docs"], deps.llm, deps.settings.max_context_tokens)
        if trace is not None:
            trace.record("generate", context_chunks=len(state["docs"]), refused=result.refused,
                         confidence=result.confidence, answer=result.answer)
        return {"answer": result}

    def rewrite_node(state: AgentState) -> dict:
        if trace is not None:
            trace.record("rewrite", attempt=state["retries"] + 1)
        return {"retries": state["retries"] + 1}

    def refuse_node(state: AgentState) -> dict:
        if state["unresolved"]:
            reason = f"no filing matches {state['unresolved']}"
        else:
            grade_reason = relevance_refusal_reason(state["docs"], deps.settings.refusal_score_threshold)
            reason = f"{grade_reason} after {state['retries']} retries"
        if trace is not None:
            trace.record("refuse", reason=reason)
        return {"answer": refusal(reason)}

    def after_resolve(state: AgentState) -> str:
        return "refuse" if not state["sources"] else "retrieve"

    def after_grade(state: AgentState) -> str:
        reason = relevance_refusal_reason(state["docs"], deps.settings.refusal_score_threshold)
        if reason is None:
            return "generate"
        return "rewrite" if state["retries"] < deps.settings.agentic_max_retries else "refuse"

    g = StateGraph(AgentState)
    for name, fn in [("resolve", resolve_node), ("retrieve", retrieve_node),
                     ("generate", generate_node), ("rewrite", rewrite_node),
                     ("refuse", refuse_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "resolve")
    g.add_conditional_edges("resolve", after_resolve,
                            {"retrieve": "retrieve", "refuse": "refuse"})
    g.add_conditional_edges("retrieve", after_grade,
                            {"generate": "generate", "rewrite": "rewrite", "refuse": "refuse"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    g.add_edge("refuse", END)
    return g.compile()


def answer_agentic(question: str, deps: QueryDeps, trace: TraceRecorder | None = None) -> RAGAnswer:
    app = build_agentic_app(deps, trace=trace)
    final = app.invoke({"question": question, "sources": [], "unresolved": [],
                        "docs": [], "answer": None, "retries": 0})
    return final["answer"]
