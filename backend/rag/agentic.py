"""Agentic query path as a LangGraph StateGraph (course: 04_agentic_rag.py).

resolve -> [refuse | retrieve] -> grade -> [generate | rewrite->retrieve | refuse]
The grade step is a simple, transparent check: did retrieval return anything?
If not and retries remain, retry retrieval; otherwise refuse.
"""

from typing import Optional
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from rag.generation.generator import generate, refusal
from rag.generation.schema import RAGAnswer
from rag.query import QueryDeps, _retrieve
from rag.retrieval.entity_resolver import parse_query, resolve


class AgentState(TypedDict):
    question: str
    sources: list[str]
    unresolved: list[str]
    docs: list
    answer: Optional[RAGAnswer]
    retries: int


def build_agentic_app(deps: QueryDeps):
    def resolve_node(state: AgentState) -> dict:
        entities = parse_query(state["question"], deps.llm)
        res = resolve(entities, deps.entity_index)
        return {"sources": res.sources, "unresolved": res.unresolved}

    def retrieve_node(state: AgentState) -> dict:
        docs = _retrieve(state["question"], "hybrid", state["sources"], deps)
        return {"docs": docs}

    def generate_node(state: AgentState) -> dict:
        return {"answer": generate(state["question"], state["docs"], deps.llm)}

    def rewrite_node(state: AgentState) -> dict:
        return {"retries": state["retries"] + 1}

    def refuse_node(state: AgentState) -> dict:
        return {"answer": refusal(f"no filing matches {state['unresolved']}")}

    def after_resolve(state: AgentState) -> str:
        return "refuse" if not state["sources"] else "retrieve"

    def after_grade(state: AgentState) -> str:
        if state["docs"]:
            return "generate"
        return "rewrite" if state["retries"] < deps.settings.max_retries else "refuse"

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


def answer_agentic(question: str, deps: QueryDeps) -> RAGAnswer:
    app = build_agentic_app(deps)
    final = app.invoke({"question": question, "sources": [], "unresolved": [],
                        "docs": [], "answer": None, "retries": 0})
    return final["answer"]
