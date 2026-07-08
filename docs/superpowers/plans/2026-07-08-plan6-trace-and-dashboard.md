# Plan 6 — Pipeline Trace + Demo Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the system's reasoning visible for a live interview demo — not just the final answer, but *how it got there*: parsed entities, which filing was resolved (or why it refused), the retrieval candidates, the reranked chunks with scores, and the generation outcome. Plus a page showing the Plan 5 benchmark results (per-category × per-mode accuracy). Two halves: (1) a backend trace mechanism threaded through the *existing* query pipeline (no parallel "demo path" — what you see is exactly what ran), and (2) a minimal React frontend that renders it.

**Architecture:**
- **Backend:** `rag/trace.py` adds an optional `TraceRecorder` that `rag.query.answer` / `answer_linear` / `_retrieve` / `rag.agentic.answer_agentic` accept as `trace: TraceRecorder | None = None`. When `None` (every existing caller — `/chat`, `eval.runner`), zero behavior change. When passed, each stage appends a step. `app/main.py` gets two new read-only additions: `POST /chat/trace` (runs the pipeline once with a recorder, returns the answer + its trace) and `GET /eval-results` (serves the committed `eval_results.json` from Plan 5).
- **Frontend:** `frontend/` — Vite + plain React (no TypeScript, no state-management library, no CSS framework). Two tabs: **Chat** (ask a question, see the answer, expand the pipeline trace) and **Eval Results** (the benchmark table).

**Tech Stack:** Backend: stdlib + existing deps, no new packages. Frontend: `vite`, `react`, `react-dom` only.

## Global Constraints

- `trace=None` must be a true no-op: no new allocations, no behavior change, on every existing call site (`/chat`, `eval.runner.run_mode`). This is why trace is threaded through the *same* functions rather than a parallel "traced" copy — one code path, no drift between what the demo shows and what production actually does.
- Chunk text in trace payloads is truncated to 200 chars (`snippet`) — enough to show relevance, not enough to bloat the response.
- `POST /chat/trace` is demo/debug tooling: no response cache, no retry/fallback chain, not counted in `/metrics`. A demo should show the real single-attempt pipeline, not a retried/cached one. Keep it deliberately simpler than `/chat`.
- This repo has no JS test runner. Backend tasks are TDD (`uv run pytest -q` from `backend/`); frontend tasks are "implement, then verify manually" with an explicit checklist per task.
- Frontend: plain `.jsx`, `useState`/`useEffect` only, no Redux/Zustand/Tailwind/MUI. Fetch API directly, no axios.
- Commit messages carry no AI-attribution trailers. Every backend task ends green (`uv run pytest -q`) and is committed; every frontend task ends with the manual checklist passing and is committed.

---

### Task 1: `TraceRecorder`

**Files:**
- Create: `backend/rag/trace.py`
- Test: `backend/tests/test_trace.py`

**Interfaces:**
- Produces:
  - `TraceStep` (dataclass): `stage: str`, `elapsed_ms: float`, `data: dict`.
  - `TraceRecorder`: `.steps: list[TraceStep]`; `.record(stage: str, **data) -> None` appends a step with elapsed time since the recorder was constructed.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_trace.py`:
```python
"""TraceRecorder: records ordered steps with elapsed time."""

from rag.trace import TraceRecorder


def test_record_appends_steps_in_order():
    t = TraceRecorder()
    t.record("parse_query", companies=["Petra"], fiscal_year="2022")
    t.record("resolve_entities", sources=["petra.pdf"], unresolved=[])

    assert len(t.steps) == 2
    assert t.steps[0].stage == "parse_query"
    assert t.steps[0].data == {"companies": ["Petra"], "fiscal_year": "2022"}
    assert t.steps[1].stage == "resolve_entities"
    assert t.steps[0].elapsed_ms >= 0
    assert t.steps[1].elapsed_ms >= t.steps[0].elapsed_ms
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_trace.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `backend/rag/trace.py`**

```python
"""Optional pipeline trace for the demo dashboard.

TraceRecorder is threaded through the query pipeline as trace=None by
default — every existing caller (/chat, eval.runner) is unaffected. Only
/chat/trace passes one in, so the recording overhead only exists when
someone is actually asking to see it.
"""

import time
from dataclasses import dataclass, field


@dataclass
class TraceStep:
    stage: str
    elapsed_ms: float
    data: dict


@dataclass
class TraceRecorder:
    steps: list[TraceStep] = field(default_factory=list)
    _start: float = field(default_factory=time.perf_counter)

    def record(self, stage: str, **data) -> None:
        elapsed = (time.perf_counter() - self._start) * 1000
        self.steps.append(TraceStep(stage=stage, elapsed_ms=round(elapsed, 2), data=data))
```

- [ ] **Step 4: Run to verify pass, then full suite**

Run: `uv run pytest tests/test_trace.py -v` → pass. Then `uv run pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add backend/rag/trace.py backend/tests/test_trace.py
git commit -m "feat(trace): add TraceRecorder for optional pipeline tracing"
```

---

### Task 2: Thread trace through the linear query path

**Files:**
- Modify: `backend/rag/query.py`
- Modify: `backend/tests/test_query.py`

**Interfaces:**
- Changes (all new params default `None` — fully backward compatible):
  - `_retrieve(question, mode, sources, deps, trace=None)` — records `"retrieve_candidates"` (pre-rerank: `{"candidates": [{"source", "page", "snippet"}]}`) and, if a reranker ran, `"rerank"` (post-rerank: `{"chunks": [{"source", "page", "score", "snippet"}]}`).
  - `answer_linear(question, mode, deps, trace=None)` — records `"parse_query"` (`{"companies", "fiscal_year"}`), `"resolve_entities"` (`{"sources", "unresolved"}`), and either `"refuse"` (`{"reason"}`) or `"generate"` (`{"context_chunks", "refused", "confidence", "answer"}`).
  - `answer(question, mode, deps, trace=None)` — passes `trace` through to whichever path it dispatches to.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_query.py` (reuse the existing `_deps` helper and `_LLM` fake already in that file):
```python
from rag.trace import TraceRecorder


def test_answer_linear_records_trace_on_success(tmp_path):
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2022")
    trace = TraceRecorder()
    answer_linear("assets of CrossFirst Bank in 2022?", "hybrid", deps, trace=trace)

    stages = [s.stage for s in trace.steps]
    assert stages == ["parse_query", "resolve_entities", "retrieve_candidates", "generate"]
    assert trace.steps[0].data["companies"] == ["CrossFirst Bank"]
    assert trace.steps[-1].data["answer"] == "Total assets were 5B."


def test_answer_linear_records_trace_on_entity_refusal(tmp_path):
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2023")
    trace = TraceRecorder()
    answer_linear("assets of CrossFirst Bank in 2023?", "hybrid", deps, trace=trace)

    stages = [s.stage for s in trace.steps]
    assert stages == ["parse_query", "resolve_entities", "refuse"]
    assert "CrossFirst Bank" in trace.steps[-1].data["reason"]


def test_answer_without_trace_is_unaffected(tmp_path):
    # trace defaults to None — this must behave exactly as before (no crash, no extra work).
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2022")
    result = answer_linear("assets of CrossFirst Bank in 2022?", "hybrid", deps)
    assert result.refused is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_query.py -v` → FAIL (`TypeError: unexpected keyword argument 'trace'`).

- [ ] **Step 3: Implement in `backend/rag/query.py`**

Add near the top (after the existing imports):
```python
from rag.trace import TraceRecorder
```

Replace `_retrieve`:
```python
def _doc_snippet(d, with_score: bool = False) -> dict:
    out = {"source": d.metadata.get("source", ""), "page": d.metadata.get("page"),
           "snippet": d.page_content[:200]}
    if with_score:
        out["score"] = d.metadata.get("rerank_score")
    return out


def _retrieve(question, mode, sources, deps, trace: TraceRecorder | None = None):
    s = deps.settings
    if mode == "basic":
        retriever = vector_only_retriever(deps.store, sources, s.top_k)
    else:  # hybrid
        retriever = build_hybrid_retriever(deps.store, deps.docstore_docs, sources,
                                           s.top_k, s.bm25_weight, s.vector_weight)
    candidates = retriever.invoke(question)
    if trace is not None:
        trace.record("retrieve_candidates", candidates=[_doc_snippet(d) for d in candidates])
    if deps.reranker is None:
        return candidates
    reranked = rerank(question, candidates, deps.reranker, s.top_n)
    if trace is not None:
        trace.record("rerank", chunks=[_doc_snippet(d, with_score=True) for d in reranked])
    return reranked
```

Replace `answer_linear`:
```python
def answer_linear(question: str, mode: str, deps: QueryDeps,
                  trace: TraceRecorder | None = None) -> RAGAnswer:
    entities = parse_query(question, deps.llm)
    if trace is not None:
        trace.record("parse_query", companies=entities.companies, fiscal_year=entities.fiscal_year)
    res = resolve(entities, deps.entity_index)
    if trace is not None:
        trace.record("resolve_entities", sources=res.sources, unresolved=res.unresolved)
    if res.refuse:
        reason = f"no filing matches {res.unresolved}"
        if trace is not None:
            trace.record("refuse", reason=reason)
        return refusal(reason)
    docs = _retrieve(question, mode, res.sources, deps, trace=trace)
    reason = relevance_refusal_reason(docs, deps.settings.refusal_score_threshold)
    if reason:
        if trace is not None:
            trace.record("refuse", reason=reason)
        return refusal(reason)
    result = generate(question, docs, deps.llm, deps.settings.max_context_tokens)
    if trace is not None:
        trace.record("generate", context_chunks=len(docs), refused=result.refused,
                     confidence=result.confidence, answer=result.answer)
    return result
```

Replace `answer`:
```python
def answer(question: str, mode: str, deps: QueryDeps,
          trace: TraceRecorder | None = None) -> RAGAnswer:
    if mode == "agentic":
        from rag.agentic import answer_agentic
        return answer_agentic(question, deps, trace=trace)
    return answer_linear(question, mode, deps, trace=trace)
```

- [ ] **Step 4: Run to verify pass, then full suite**

Run: `uv run pytest tests/test_query.py -v` → all pass. Then `uv run pytest -q` → green (this will fail until Task 3 also updates `answer_agentic`'s signature — if so, do Tasks 2 and 3 together before running the full suite).

- [ ] **Step 5: Commit**

```bash
git add backend/rag/query.py backend/tests/test_query.py
git commit -m "feat(trace): thread optional TraceRecorder through the linear query path"
```

---

### Task 3: Thread trace through the agentic path

**Files:**
- Modify: `backend/rag/agentic.py`
- Modify: `backend/tests/test_agentic.py`

**Interfaces:**
- Changes (all new params default `None`):
  - `build_agentic_app(deps, trace=None)`, `answer_agentic(question, deps, trace=None)`.
  - Records the same `"parse_query"` / `"resolve_entities"` / `"retrieve_candidates"` / `"rerank"` / `"generate"` / `"refuse"` stages as the linear path (via the same `_retrieve` from Task 2), plus `"rewrite"` (`{"attempt": n}`) on each retry — so a demo question that needs a retry visibly shows the self-correction loop as repeated retrieve/rerank steps.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_agentic.py`:
```python
from rag.trace import TraceRecorder


def test_agentic_records_trace_on_success(tmp_path):
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2022")
    trace = TraceRecorder()
    answer_agentic("assets of CrossFirst Bank in 2022?", deps, trace=trace)

    stages = [s.stage for s in trace.steps]
    assert stages == ["parse_query", "resolve_entities", "retrieve_candidates", "generate"]


def test_agentic_records_rewrite_on_retry(tmp_path, monkeypatch):
    import rag.agentic as agentic_mod

    deps = _deps(tmp_path, ["CrossFirst Bank"], "2022")
    calls = {"n": 0}
    real_retrieve = agentic_mod._retrieve

    def flaky_retrieve(question, mode, sources, deps, trace=None):
        calls["n"] += 1
        if calls["n"] == 1:
            if trace is not None:
                trace.record("retrieve_candidates", candidates=[])
            return []  # first attempt: nothing found -> triggers a retry
        return real_retrieve(question, mode, sources, deps, trace=trace)

    monkeypatch.setattr(agentic_mod, "_retrieve", flaky_retrieve)

    trace = TraceRecorder()
    result = answer_agentic("assets of CrossFirst Bank in 2022?", deps, trace=trace)

    assert result.refused is False
    stages = [s.stage for s in trace.steps]
    assert "rewrite" in stages
    assert stages.count("retrieve_candidates") == 2  # first (empty) + retry
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_agentic.py -v` → FAIL (`TypeError: unexpected keyword argument 'trace'`).

- [ ] **Step 3: Implement in `backend/rag/agentic.py`**

Update the imports and every function in `build_agentic_app` / `answer_agentic`:
```python
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
```

Note: only `build_agentic_app`'s signature, `retrieve_node`, `generate_node`, `rewrite_node`, `refuse_node`, `resolve_node`, and `answer_agentic`'s signature actually change — `after_resolve`/`after_grade`/the graph wiring are unchanged, shown above for context only.

- [ ] **Step 4: Run to verify pass, then full suite**

Run: `uv run pytest tests/test_agentic.py -v` → all pass. Then `uv run pytest -q` → full suite green.

- [ ] **Step 5: Commit**

```bash
git add backend/rag/agentic.py backend/tests/test_agentic.py
git commit -m "feat(trace): thread optional TraceRecorder through the agentic loop"
```

---

### Task 4: `POST /chat/trace` and `GET /eval-results`

**Files:**
- Modify: `backend/app/models.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_api.py`

**Interfaces:**
- Produces (in `app/models.py`):
  - `TraceStepModel(BaseModel)`: `stage: str`, `elapsed_ms: float`, `data: dict`.
  - `TraceResponse(ChatResponse)`: adds `steps: list[TraceStepModel] = Field(default_factory=list)`.
- Adds (in `app/main.py`):
  - `POST /chat/trace` — same request body as `/chat` (`ChatRequest`); screens input the same way; runs `answer(cleaned, mode, query_deps, trace=recorder)` **once, no retry, no fallback, no cache**; masks output the same way; returns `TraceResponse`. On exception, `500` with the same `ErrorResponse` shape as `/chat`.
  - `GET /eval-results` — reads `Path(settings.data_dir).parent / "eval_results.json"` (the file Plan 5's CLI writes); `404` with `{"error": "no eval run yet — run `make eval`"}` if it doesn't exist yet; otherwise returns its parsed JSON directly.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_api.py`:
```python
import json


def test_chat_trace_returns_steps(client):
    r = client.post("/chat/trace", json={"message": "net income of Petra 2022?"})
    assert r.status_code == 200
    body = r.json()
    assert body["response"] == "88.1"
    assert "steps" in body
    # fake_answer (the client fixture's monkeypatched `answer`) doesn't record
    # anything itself, so this only proves the endpoint wires the field through -
    # real step content is covered by test_query.py / test_agentic.py.
    assert isinstance(body["steps"], list)


def test_chat_trace_not_cached_or_retried(client):
    # boom always raises in fake_answer - /chat/trace has no retry/fallback,
    # so it should surface as a single clean 500, not loop or hang.
    r = client.post("/chat/trace", json={"message": "boom"})
    assert r.status_code == 500


def test_eval_results_404_when_missing(client, tmp_path, monkeypatch):
    r = client.get("/eval-results")
    assert r.status_code == 404


def test_eval_results_returns_report(client, monkeypatch, tmp_path):
    from core.config import get_settings
    settings = get_settings()
    eval_path = tmp_path / "eval_results.json"
    eval_path.write_text(json.dumps({"modes": {"hybrid": {"overall": {"accuracy": 1.0}}}}),
                         encoding="utf-8")
    monkeypatch.setattr(settings, "data_dir", str(tmp_path / "docs"))
    r = client.get("/eval-results")
    assert r.status_code == 200
    assert r.json()["modes"]["hybrid"]["overall"]["accuracy"] == 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_api.py -v` → FAIL (`404 Not Found` on `/chat/trace`, etc).

- [ ] **Step 3: Implement**

In `backend/app/models.py`, add after `ChatResponse`:
```python
class TraceStepModel(BaseModel):
    stage: str
    elapsed_ms: float
    data: dict


class TraceResponse(ChatResponse):
    steps: list[TraceStepModel] = Field(default_factory=list)
```

In `backend/app/main.py`, add imports:
```python
import json
from pathlib import Path

from app.models import TraceResponse, TraceStepModel
from rag.trace import TraceRecorder
```

Add the two routes (after the existing `/chat` route, before `return app`):
```python
    @app.post("/chat/trace", response_model=TraceResponse)
    @limiter.limit(settings.rate_limit)
    def chat_trace(request: Request, body: ChatRequest):
        state = request.app.state
        started = time.perf_counter()
        mode = body.mode or settings.retrieval_mode

        blocked, cleaned = screen_input(body.message)
        if blocked:
            return JSONResponse(status_code=400,
                                content=ErrorResponse(error="input rejected by safety filter").model_dump())

        query_deps = _scoped_deps(state.query_deps, body.top_k, body.top_n)
        recorder = TraceRecorder()
        try:
            # Demo/debug endpoint: one real attempt, no retry/fallback/cache -
            # what you see here is exactly what ran, not a hidden second try.
            rag_answer = answer(cleaned, mode, query_deps, trace=recorder)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(status_code=500,
                                content=ErrorResponse(error="generation failed",
                                                      detail=str(exc)).model_dump())

        rag_answer.answer = mask_output(rag_answer.answer)
        elapsed = (time.perf_counter() - started) * 1000
        base = to_chat_response(rag_answer, thread_id=body.thread_id,
                                model_used=settings.primary_model, mode=mode,
                                cached=False, processing_time_ms=elapsed)
        return TraceResponse(**base.model_dump(),
                             steps=[TraceStepModel(stage=s.stage, elapsed_ms=s.elapsed_ms, data=s.data)
                                    for s in recorder.steps])

    @app.get("/eval-results")
    def eval_results(request: Request):
        path = Path(settings.data_dir).parent / "eval_results.json"
        if not path.exists():
            return JSONResponse(status_code=404,
                                content={"error": "no eval run yet - run `make eval`"})
        return json.loads(path.read_text(encoding="utf-8"))
```

- [ ] **Step 4: Run to verify pass, then full suite**

Run: `uv run pytest tests/test_api.py -v` → all pass. Then `uv run pytest -q` → full suite green.

- [ ] **Step 5: Commit**

```bash
git add backend/app/models.py backend/app/main.py backend/tests/test_api.py
git commit -m "feat(api): add /chat/trace (pipeline trace) and /eval-results (benchmark table)"
```

---

### Task 5: Frontend scaffold + Chat page + Trace panel

**Files:**
- Create: `frontend/package.json`, `frontend/vite.config.js`, `frontend/index.html`
- Create: `frontend/src/main.jsx`, `frontend/src/App.jsx`, `frontend/src/api.js`, `frontend/src/styles.css`
- Create: `frontend/src/components/ChatPage.jsx`, `frontend/src/components/TracePanel.jsx`

No pytest here — this task's "test" is running the dev server and manually verifying against the real API (Task 4's endpoints, backed by Tasks 1-3's real trace data).

- [ ] **Step 1: Scaffold**

`frontend/package.json`:
```json
{
  "name": "financial-rag-frontend",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.1",
    "vite": "^5.4.0"
  }
}
```

`frontend/vite.config.js`:
```js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
});
```

`frontend/index.html`:
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Financial-Report RAG</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

`frontend/src/main.jsx`:
```jsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

- [ ] **Step 2: API client**

`frontend/src/api.js`:
```js
const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function postJson(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `request failed: ${res.status}`);
  return data;
}

export function postChatTrace({ message, mode, topK, topN }) {
  const body = { message };
  if (mode) body.mode = mode;
  if (topK) body.top_k = Number(topK);
  if (topN) body.top_n = Number(topN);
  return postJson("/chat/trace", body);
}

export async function getEvalResults() {
  const res = await fetch(`${BASE}/eval-results`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`request failed: ${res.status}`);
  return res.json();
}
```

- [ ] **Step 3: Trace panel**

`frontend/src/components/TracePanel.jsx`:
```jsx
const STAGE_LABELS = {
  parse_query: "1. Parse question -> entities",
  resolve_entities: "2. Resolve entities -> source filing(s)",
  retrieve_candidates: "3. Retrieve candidates",
  rerank: "4. Rerank",
  rewrite: "↻ Retry (weak relevance)",
  generate: "5. Generate answer",
  refuse: "✕ Refuse",
};

function StepBody({ stage, data }) {
  if (stage === "retrieve_candidates" || stage === "rerank") {
    const chunks = data.candidates || data.chunks || [];
    const hasScore = stage === "rerank";
    return (
      <table className="chunk-table">
        <thead>
          <tr>
            <th>source</th><th>page</th>{hasScore && <th>score</th>}<th>snippet</th>
          </tr>
        </thead>
        <tbody>
          {chunks.map((c, i) => (
            <tr key={i}>
              <td>{c.source}</td>
              <td>{c.page}</td>
              {hasScore && <td>{c.score != null ? c.score.toFixed(4) : "-"}</td>}
              <td className="snippet">{c.snippet}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }
  return <pre className="step-data">{JSON.stringify(data, null, 2)}</pre>;
}

export default function TracePanel({ steps }) {
  if (!steps || steps.length === 0) return null;
  return (
    <div className="trace-panel">
      <p className="trace-heading">Pipeline trace ({steps.length} steps)</p>
      <div className="trace-steps">
        {steps.map((s, i) => (
          <details key={i} open={i === steps.length - 1}>
            <summary>{STAGE_LABELS[s.stage] || s.stage} - {s.elapsed_ms}ms</summary>
            <StepBody stage={s.stage} data={s.data} />
          </details>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Chat page**

`frontend/src/components/ChatPage.jsx`:
```jsx
import { useState } from "react";
import { postChatTrace } from "../api.js";
import TracePanel from "./TracePanel.jsx";

export default function ChatPage() {
  const [message, setMessage] = useState("");
  const [mode, setMode] = useState("");
  const [topK, setTopK] = useState("");
  const [topN, setTopN] = useState("");
  const [history, setHistory] = useState([]); // newest first: [{question, response}]
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleAsk(e) {
    e.preventDefault();
    if (!message.trim() || loading) return;
    setLoading(true);
    setError("");
    try {
      const response = await postChatTrace({ message, mode, topK, topN });
      setHistory((h) => [{ question: message, response }, ...h]);
      setMessage("");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="chat-page">
      <form onSubmit={handleAsk} className="chat-form">
        <input
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder='e.g. What was the net income of "Petra Diamonds" in fiscal year 2022?'
        />
        <select value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="">default mode</option>
          <option value="basic">basic</option>
          <option value="hybrid">hybrid</option>
          <option value="agentic">agentic</option>
        </select>
        <input value={topK} onChange={(e) => setTopK(e.target.value)} placeholder="top_k" className="num" />
        <input value={topN} onChange={(e) => setTopN(e.target.value)} placeholder="top_n" className="num" />
        <button type="submit" disabled={loading}>{loading ? "Asking..." : "Ask"}</button>
      </form>
      {error && <p className="error">{error}</p>}
      <div className="history">
        {history.map((h, i) => (
          <div key={i} className="qa-block">
            <p className="question">Q: {h.question}</p>
            <p className={"answer" + (h.response.refused ? " refused" : "")}>
              A: {h.response.response}
              {h.response.refused && h.response.refusal_reason && (
                <span className="reason"> ({h.response.refusal_reason})</span>
              )}
            </p>
            {h.response.citations && h.response.citations.length > 0 && (
              <div className="citations">
                {h.response.citations.map((c, j) => (
                  <span key={j} className="chip">{c.source} p{c.page}</span>
                ))}
              </div>
            )}
            <TracePanel steps={h.response.steps} />
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 5: `App.jsx` (single tab for now — Task 6 adds the second)**

`frontend/src/App.jsx`:
```jsx
import ChatPage from "./components/ChatPage.jsx";

export default function App() {
  return (
    <div className="app">
      <h1>Financial-Report RAG</h1>
      <ChatPage />
    </div>
  );
}
```

- [ ] **Step 6: Minimal styles**

`frontend/src/styles.css` — keep it plain: a monospace/system font, a max-width centered column, simple borders for the trace table and citation chips, `.refused { color: #b00020 }`. No frameworks; a few dozen lines is enough.

- [ ] **Step 7: Manual verification checklist**

1. `cd frontend && npm install && npm run dev` (or `make ui` once Task 7 wires it).
2. Backend running separately (`make run` from `backend/`) with a built index.
3. Ask `net income of Petra 2022?` → see `88.1`, citations, and an expandable trace with `parse_query -> resolve_entities -> retrieve_candidates -> rerank -> generate`.
4. Ask a wrong-year question (`net income of CrossFirst Bank 2023?`) → see the refusal + reason, and a trace ending in `refuse`.
5. Switch mode to `agentic` and re-ask → trace still renders (may include a `rewrite` step if a retry happens).
6. Confirm the browser console has no errors.

- [ ] **Step 8: Commit**

```bash
git add frontend/
git commit -m "feat(ui): Vite/React chat page with pipeline trace panel"
```

---

### Task 6: Eval results page

**Files:**
- Create: `frontend/src/components/EvalResultsPage.jsx`
- Modify: `frontend/src/App.jsx`

- [ ] **Step 1: Implement**

`frontend/src/components/EvalResultsPage.jsx`:
```jsx
import { useEffect, useState } from "react";
import { getEvalResults } from "../api.js";

export default function EvalResultsPage() {
  const [report, setReport] = useState(null);
  const [notFound, setNotFound] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getEvalResults()
      .then((r) => (r ? setReport(r) : setNotFound(true)))
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <p className="error">{error}</p>;
  if (notFound) return <p className="eval-empty">No eval run yet - run `make eval` in backend/.</p>;
  if (!report) return <p>Loading...</p>;

  const modes = Object.keys(report.modes);
  const categories = [...new Set(modes.flatMap((m) => Object.keys(report.modes[m].by_category)))].sort();

  return (
    <div className="eval-page">
      <h2>Benchmark results (40-question ERC eval)</h2>
      <table className="eval-table">
        <thead>
          <tr><th>category</th>{modes.map((m) => <th key={m}>{m}</th>)}</tr>
        </thead>
        <tbody>
          {categories.map((cat) => (
            <tr key={cat}>
              <td>{cat}</td>
              {modes.map((m) => {
                const s = report.modes[m].by_category[cat];
                return (
                  <td key={m}>
                    {s ? `${Math.round(s.accuracy * 100)}% (${s.correct}/${s.total})` : "-"}
                  </td>
                );
              })}
            </tr>
          ))}
          <tr className="overall-row">
            <td>OVERALL</td>
            {modes.map((m) => <td key={m}>{Math.round(report.modes[m].overall.accuracy * 100)}%</td>)}
          </tr>
          <tr className="refusal-row">
            <td>refusal accuracy</td>
            {modes.map((m) => <td key={m}>{Math.round(report.refusal_accuracy[m] * 100)}%</td>)}
          </tr>
        </tbody>
      </table>
    </div>
  );
}
```

Update `frontend/src/App.jsx` to add a tab switcher:
```jsx
import { useState } from "react";
import ChatPage from "./components/ChatPage.jsx";
import EvalResultsPage from "./components/EvalResultsPage.jsx";

export default function App() {
  const [tab, setTab] = useState("chat");
  return (
    <div className="app">
      <h1>Financial-Report RAG</h1>
      <div className="tabs">
        <button className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}>Chat</button>
        <button className={tab === "eval" ? "active" : ""} onClick={() => setTab("eval")}>Eval Results</button>
      </div>
      {tab === "chat" ? <ChatPage /> : <EvalResultsPage />}
    </div>
  );
}
```

- [ ] **Step 2: Manual verification checklist**

1. Without having run `make eval` yet: open the Eval Results tab → see the "no eval run yet" message, not a crash.
2. After running Plan 5's `make eval` (from `backend/`) at least once: reload the tab → see the per-category × per-mode table and the refusal-accuracy row, numbers matching the CLI's printed table.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/EvalResultsPage.jsx frontend/src/App.jsx
git commit -m "feat(ui): eval results tab showing the Plan 5 benchmark table"
```

---

### Task 7: Wire `make ui`, README quickstart, final review

**Files:**
- Modify: `backend/Makefile` (confirm/fix the `ui` target — it should `cd ../frontend && npm run dev`)
- Modify: `README.md` (add the frontend quickstart + a screenshot-free description of the trace panel and eval tab)

- [ ] **Step 1: Confirm/fix the Makefile `ui` target**

In `backend/Makefile`:
```make
ui:
	cd ../frontend && npm install && npm run dev
```

- [ ] **Step 2: README quickstart addition**

Add a short section: `make run` (API) in one terminal, `make ui` (frontend) in another, open `http://localhost:5173`. One sentence each on the Chat tab (ask a question, expand the trace to see entity resolution -> retrieval -> reranking -> generation) and the Eval Results tab (requires `make eval` to have been run at least once).

- [ ] **Step 3: Full-system manual walkthrough**

1. `cd backend && make run` (terminal 1).
2. `cd backend && make ui` (terminal 2) — or `cd frontend && npm run dev` if the Makefile target isn't used.
3. Open `http://localhost:5173`. Ask 2-3 questions covering: a normal answerable question, a wrong-year refusal, an unknown-company refusal. Confirm the trace panel is coherent for each (refusals show a short trace ending in `refuse`; answers show the full `parse_query -> resolve_entities -> retrieve_candidates -> rerank -> generate` chain).
4. Run `cd backend && make eval` once (needs `GROQ_API_KEY`), then reload the Eval Results tab and confirm the table populates.
5. `uv run pytest -q` from `backend/` → full suite green (should be unchanged in count from Task 4 plus Tasks 1-3's new tests — no frontend tests exist to run).

- [ ] **Step 4: Commit**

```bash
git add backend/Makefile README.md
git commit -m "docs: wire make ui and document the dashboard quickstart"
```

---

## Self-Review

**Spec coverage (user's stated goal — "show the reasoning process, how the request is processed, retrieved chunks, reranking result"):**
- Reasoning process / request flow visualized step-by-step (entity parse -> resolve -> retrieve -> rerank -> generate/refuse), including agentic retries → Tasks 1-3, 5. ✅
- Retrieved chunks (pre-rerank) and reranking result (post-rerank, with scores) both shown, not collapsed into one step → Task 2's `_retrieve` records both stages separately. ✅
- Refusal reasoning surfaced (reuses `RAGAnswer.reason` from the earlier refusal-reason work) → Task 4. ✅
- Benchmark performance visible for the demo (Plan 5's `eval_results.json`) → Tasks 4, 6. ✅
- One pipeline, not two: `trace=None` is a true no-op on every existing call site (`/chat`, `eval.runner`) — the demo shows exactly what production runs, not a simplified stand-in → Global Constraints, Tasks 2-3.

**Placeholder scan:** none — full code in every backend step; frontend steps have full component code with a manual (not automated) acceptance checklist, since this repo has no JS test runner.

**Type/interface consistency:** `TraceRecorder`/`TraceStep` (Task 1) consumed by `_retrieve`/`answer_linear`/`answer` (Task 2) and `build_agentic_app`/`answer_agentic` (Task 3) — same `.record(stage, **data)` call shape throughout. `TraceStepModel`/`TraceResponse` (Task 4) mirror the recorder's `stage`/`elapsed_ms`/`data` fields exactly, built via `to_chat_response(...)` reuse (no duplicated response-assembly logic) plus `steps=[...]`. Frontend `TracePanel` (Task 5) renders exactly the `stage`/`data` shapes Tasks 2-3 produce (`retrieve_candidates` -> `data.candidates`, `rerank` -> `data.chunks`, both `{source, page, snippet}` + optional `score`) — verified against the plan's own backend code, not assumed.

**Backward compatibility:** every changed function signature (`_retrieve`, `answer_linear`, `answer`, `build_agentic_app`, `answer_agentic`) adds `trace` as the last parameter with default `None` — `eval.runner.run_mode`'s positional 3-arg call (`answer_fn(item.question, mode, deps)`) and `/chat`'s existing call are both unaffected.
