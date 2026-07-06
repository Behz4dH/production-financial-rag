# Plan 3 — Retrieval & Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the online query path — parse a question into entities, resolve them to source filings (or refuse), retrieve with a hybrid (BM25 ⊕ vector) retriever filtered to those sources, and generate a grounded, cited answer (or `N/A`) — exposed as one `answer(question, mode)` function.

**Architecture:** Standard LangChain/LangGraph idioms only (as used in `production-course-main-code`): `with_structured_output` for query parsing + the answer, `EnsembleRetriever`+`BM25Retriever` for hybrid retrieval (`advanced_rag.py`), a small transparent cross-encoder reranker (kept simple — a scoring function, not a compression chain), `StateGraph`+`add_conditional_edges` for the agentic mode (`04_agentic_rag.py`, `langgraph_core.py`), `init_chat_model` for the LLM.

**Tech Stack:** `langchain` (`init_chat_model`, `with_structured_output`), `langchain-classic` (`EnsembleRetriever`), `langchain-community` (`BM25Retriever`), `langchain-chroma`, `langgraph`, `pydantic`, `pytest`.

## Global Constraints

- The `rag/` package stays framework-free of the web layer — it must NOT import from `app/`. Shared answer/citation types live in `rag/generation/schema.py`.
- All env/config via `core.config.get_settings()`; no other module reads env vars.
- **Refusal is the headline behavior:** if the asked company is not in `doc_metadata.json`, or its filing's `fiscal_year` ≠ the asked year, the system returns `N/A` **before generating** — never a fabricated number.
- Entity attributes (company/year/currency) are read from `doc_metadata.json` at query time (single source of truth). Chunks carry only `{source, page, chunk_id}` — do NOT reintroduce entity fields onto chunks.
- Tests run with NO API key / network / model download: the LLM and embeddings are mocked or faked. Real Chroma + BM25 run locally on tiny fixtures. NO repo-wide `conftest.py`.
- Keep code direct and readable — no speculative abstraction. Commit messages: no AI attribution trailers.
- Every task ends green (`uv run pytest -q`) and is committed. Work from `backend/`.

---

### Task 1: Docstore loader

**Files:**
- Create: `backend/rag/retrieval/docstore.py`
- Test: `backend/tests/test_docstore.py`

**Interfaces:**
- Consumes: `docstore.jsonl` (one JSON per chunk: `{page_content, metadata}`), written by Plan 2's pipeline.
- Produces: `load_docstore(path: str) -> list[Document]` — the corpus of chunk Documents that BM25 indexes.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_docstore.py`:
```python
"""Load the persisted chunk docstore into Documents (for BM25)."""

import json

from langchain_core.documents import Document

from rag.retrieval.docstore import load_docstore


def test_load_docstore_reads_documents(tmp_path):
    p = tmp_path / "docstore.jsonl"
    rows = [
        {"page_content": "total assets were five billion",
         "metadata": {"source": "a.pdf", "page": 3, "chunk_id": "a.pdf::p3::c0"}},
        {"page_content": "net income eighty eight million",
         "metadata": {"source": "b.pdf", "page": 4, "chunk_id": "b.pdf::p4::c0"}},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    docs = load_docstore(str(p))
    assert len(docs) == 2
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "total assets were five billion"
    assert docs[0].metadata["source"] == "a.pdf"


def test_load_docstore_missing_file_returns_empty(tmp_path):
    assert load_docstore(str(tmp_path / "nope.jsonl")) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_docstore.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `backend/rag/retrieval/docstore.py`**

```python
"""Load the persisted chunk docstore (docstore.jsonl) into Documents."""

import json
from pathlib import Path

from langchain_core.documents import Document


def load_docstore(path: str) -> list[Document]:
    p = Path(path)
    if not p.exists():
        return []
    docs: list[Document] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        docs.append(Document(page_content=row["page_content"], metadata=row["metadata"]))
    return docs
```

- [ ] **Step 4: Run to verify pass, then full suite**

Run: `uv run pytest tests/test_docstore.py -v` → 2 pass. Then `uv run pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add backend/rag/retrieval/docstore.py backend/tests/test_docstore.py
git commit -m "feat: load persisted chunk docstore for BM25"
```

---

### Task 2: Entity resolver (parse + match → sources or refuse)

**Files:**
- Create: `backend/rag/retrieval/entity_resolver.py`
- Test: `backend/tests/test_entity_resolver.py`

**Interfaces:**
- Consumes: an `LLM` (query parsing), `doc_metadata.json`.
- Produces:
  - `QueryEntities(BaseModel)`: `companies: list[str]`, `fiscal_year: str | None`.
  - `parse_query(question: str, llm) -> QueryEntities` — `llm.with_structured_output(QueryEntities)`.
  - `build_index(doc_metadata: dict) -> list[dict]` — flat records `{source, name, fiscal_year}` (one per company_name + alias) for matching.
  - `Resolution` (dataclass): `sources: list[str]`, `unresolved: list[str]`; property `refuse -> bool` (True when `sources` empty).
  - `resolve(entities: QueryEntities, index: list[dict]) -> Resolution` — for each asked company, match a record by normalized name AND (year matches, or no year asked); collect sources; unmatched/wrong-year companies go to `unresolved`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_entity_resolver.py`:
```python
"""Entity resolution: name/year matching (pure) + LLM query parse (mocked)."""

from rag.retrieval import entity_resolver as R
from rag.retrieval.entity_resolver import QueryEntities


_META = {
    "cross.pdf": {"company_name": "CrossFirst Bankshares, Inc.", "aliases": ["CrossFirst Bank"],
                  "fiscal_year": "2022"},
    "tru.pdf": {"company_name": "TransUnion", "aliases": [], "fiscal_year": "2022"},
}


def _index():
    return R.build_index(_META)


def test_index_includes_aliases():
    names = {r["name"] for r in _index()}
    assert "CrossFirst Bankshares, Inc." in names
    assert "CrossFirst Bank" in names  # alias indexed


def test_resolve_matches_alias_and_correct_year():
    e = QueryEntities(companies=["CrossFirst Bank"], fiscal_year="2022")
    res = R.resolve(e, _index())
    assert res.sources == ["cross.pdf"]
    assert res.unresolved == []
    assert res.refuse is False


def test_resolve_wrong_year_is_unresolved_refusal():
    e = QueryEntities(companies=["CrossFirst Bank"], fiscal_year="2023")
    res = R.resolve(e, _index())
    assert res.sources == []
    assert res.unresolved == ["CrossFirst Bank"]
    assert res.refuse is True


def test_resolve_unknown_company_refuses():
    e = QueryEntities(companies=["Nonexistent Corp"], fiscal_year="2022")
    res = R.resolve(e, _index())
    assert res.refuse is True
    assert res.unresolved == ["Nonexistent Corp"]


def test_resolve_no_year_asked_matches_on_name():
    e = QueryEntities(companies=["TransUnion"], fiscal_year=None)
    res = R.resolve(e, _index())
    assert res.sources == ["tru.pdf"]


def test_parse_query_uses_structured_output():
    class _Structured:
        def invoke(self, _p):
            return QueryEntities(companies=["TransUnion"], fiscal_year="2022")

    class _LLM:
        def with_structured_output(self, _schema):
            return _Structured()

    parsed = R.parse_query('total assets of "TransUnion" in 2022?', _LLM())
    assert parsed.companies == ["TransUnion"]
    assert parsed.fiscal_year == "2022"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_entity_resolver.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `backend/rag/retrieval/entity_resolver.py`**

```python
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
        hit = None
        for rec in index:
            if _matches(company, rec["name"]):
                year_ok = entities.fiscal_year is None or rec["fiscal_year"] == entities.fiscal_year
                if year_ok:
                    hit = rec["source"]
                    break
        if hit is not None:
            if hit not in res.sources:
                res.sources.append(hit)
        else:
            res.unresolved.append(company)
    return res
```

- [ ] **Step 4: Run to verify pass, then full suite**

Run: `uv run pytest tests/test_entity_resolver.py -v` → all pass. Then `uv run pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add backend/rag/retrieval/entity_resolver.py backend/tests/test_entity_resolver.py
git commit -m "feat: entity resolver (query parse + company/year -> source, refusal)"
```

---

### Task 3: Hybrid retriever (BM25 ⊕ vector, filtered by source)

**Files:**
- Create: `backend/rag/retrieval/hybrid.py`
- Test: `backend/tests/test_hybrid.py`

**Interfaces:**
- Consumes: a `ChromaStore` (Plan 2), the docstore `Document`s (Task 1).
- Produces:
  - `build_hybrid_retriever(store, docstore_docs, sources, k, bm25_weight, vector_weight)` → an `EnsembleRetriever` over only the chunks from `sources` (BM25 on the source-filtered docs + Chroma vector with a `source` filter). Built per query — `sources` is small.
  - `vector_only_retriever(store, sources, k)` → a Chroma retriever with the same `source` filter (the `basic` mode).

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_hybrid.py`:
```python
"""Hybrid retriever = BM25 (source-filtered docs) + Chroma vector (source filter)."""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.retrieval.hybrid import build_hybrid_retriever, vector_only_retriever
from rag.retrieval.store import ChromaStore


class _HashEmbeddings(Embeddings):
    def _v(self, t):
        v = [0.0] * 8
        for tok in t.lower().split():
            v[hash(tok) % 8] += 1.0
        return v
    def embed_documents(self, texts): return [self._v(t) for t in texts]
    def embed_query(self, text): return self._v(text)


def _docs():
    return [
        Document(page_content="cross first total assets five billion",
                 metadata={"source": "cross.pdf", "page": 1, "chunk_id": "cross.pdf::p1::c0"}),
        Document(page_content="transunion revenue growth",
                 metadata={"source": "tru.pdf", "page": 1, "chunk_id": "tru.pdf::p1::c0"}),
    ]


def _store(tmp_path):
    store = ChromaStore(_HashEmbeddings(), str(tmp_path / "chroma"), "hybrid_test")
    store.add(_docs())
    return store


def test_hybrid_restricts_to_source(tmp_path):
    store = _store(tmp_path)
    retriever = build_hybrid_retriever(store, _docs(), ["cross.pdf"], k=5,
                                       bm25_weight=0.4, vector_weight=0.6)
    results = retriever.invoke("total assets")
    assert results
    assert all(d.metadata["source"] == "cross.pdf" for d in results)


def test_vector_only_restricts_to_source(tmp_path):
    store = _store(tmp_path)
    retriever = vector_only_retriever(store, ["tru.pdf"], k=5)
    results = retriever.invoke("revenue")
    assert all(d.metadata["source"] == "tru.pdf" for d in results)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_hybrid.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `backend/rag/retrieval/hybrid.py`**

```python
"""Hybrid retrieval: weighted ensemble of BM25 (keyword) + vector (semantic),
restricted to the resolved source filing(s).

Built per query because the source set is small (usually 1-3 filings), so BM25
indexes only those filings' chunks and the vector search filters by source.
Uses LangChain's EnsembleRetriever + BM25Retriever (course: advanced_rag.py).
"""

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from rag.retrieval.store import ChromaStore


def _source_filter(sources: list[str]) -> dict:
    return {"source": {"$in": sources}} if len(sources) > 1 else {"source": sources[0]}


def vector_only_retriever(store: ChromaStore, sources: list[str], k: int):
    return store.as_retriever(k=k, filter=_source_filter(sources))


def build_hybrid_retriever(
    store: ChromaStore,
    docstore_docs: list,
    sources: list[str],
    k: int,
    bm25_weight: float,
    vector_weight: float,
) -> EnsembleRetriever:
    subset = [d for d in docstore_docs if d.metadata.get("source") in sources]
    bm25 = BM25Retriever.from_documents(subset)
    bm25.k = k
    vector = vector_only_retriever(store, sources, k)
    return EnsembleRetriever(retrievers=[bm25, vector],
                             weights=[bm25_weight, vector_weight])
```

Note: uses `ChromaStore.as_retriever(k, filter)` (added in the readability pass) — no reaching into internals. Verify the `EnsembleRetriever` import path against the installed `langchain-classic` (fallback: `langchain.retrievers`); keep the function signatures unchanged.

- [ ] **Step 4: Run to verify pass, then full suite**

Run: `uv run pytest tests/test_hybrid.py -v` → 2 pass. Then `uv run pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add backend/rag/retrieval/hybrid.py backend/tests/test_hybrid.py
git commit -m "feat: hybrid BM25+vector retriever filtered by source"
```

---

### Task 4: Simple cross-encoder reranker

**Files:**
- Create: `backend/rag/retrieval/reranker.py`
- Test: `backend/tests/test_reranker.py`

**Interfaces:**
- Produces:
  - `rerank(query, docs, model, top_n) -> list[Document]` — score each `(query, doc)` pair with a cross-encoder, return the top_n by score. A small, transparent function (no compression-chain wrapper).
  - `load_reranker(model_name)` — lazy `sentence_transformers.CrossEncoder` loader (downloads on first use; tests pass a fake model, so no download).

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_reranker.py`:
```python
"""Cross-encoder reranker ordering (fake model — no download)."""

from langchain_core.documents import Document

from rag.retrieval.reranker import rerank


class _FakeCrossEncoder:
    """Scores by keyword presence so ordering is deterministic and checkable."""
    def predict(self, pairs):
        return [2.0 if "assets" in doc.lower() else 0.1 for _q, doc in pairs]


def _docs():
    return [
        Document(page_content="revenue grew this year", metadata={"page": 1}),
        Document(page_content="total assets were five billion", metadata={"page": 2}),
        Document(page_content="board of directors", metadata={"page": 3}),
    ]


def test_rerank_orders_by_score_and_truncates():
    out = rerank("total assets", _docs(), _FakeCrossEncoder(), top_n=2)
    assert len(out) == 2
    assert out[0].metadata["page"] == 2  # the 'assets' doc ranks first


def test_rerank_empty_is_noop():
    assert rerank("q", [], _FakeCrossEncoder(), top_n=5) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_reranker.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `backend/rag/retrieval/reranker.py`**

```python
"""Cross-encoder reranking: score each (query, chunk) pair and keep the best.

Simple and transparent — a cross-encoder reads the query and chunk together and
returns a relevance score, which is more precise than the retriever's fusion
ranking for the final few chunks the LLM actually sees.
"""

from langchain_core.documents import Document


def rerank(query: str, docs: list[Document], model, top_n: int) -> list[Document]:
    if not docs:
        return docs
    scores = model.predict([(query, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]


def load_reranker(model_name: str):
    """Load the cross-encoder (downloads weights on first use)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)
```

- [ ] **Step 4: Run to verify pass, then full suite**

Run: `uv run pytest tests/test_reranker.py -v` → 2 pass. Then `uv run pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add backend/rag/retrieval/reranker.py backend/tests/test_reranker.py
git commit -m "feat: simple cross-encoder reranker"
```

---

### Task 5: Grounded generation (RAGAnswer)

**Files:**
- Create: `backend/rag/generation/__init__.py` (empty)
- Create: `backend/rag/generation/schema.py`
- Create: `backend/rag/generation/generator.py`
- Test: `backend/tests/test_generation.py`

**Interfaces:**
- Produces:
  - `schema.py`: `Citation(BaseModel)` (`source, company, fiscal_year, page`) and `RAGAnswer(BaseModel)` (`answer: str`, `citations: list[Citation] = []`, `refused: bool = False`, `confidence: str = "medium"`).
  - `generator.py`:
    - `format_context(docs) -> str` — number each chunk with its `source · page` tag.
    - `generate(question, docs, llm) -> RAGAnswer` — `llm.with_structured_output(RAGAnswer)` over a refusal-first grounded prompt; user text bound as a variable (never `.format`-ed in).
    - `refusal(reason: str) -> RAGAnswer` — a grounded `N/A` answer (`refused=True`), no LLM call.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_generation.py`:
```python
"""Grounded generation returns a structured RAGAnswer (LLM mocked)."""

from langchain_core.documents import Document

from rag.generation.generator import format_context, generate, refusal
from rag.generation.schema import RAGAnswer


def _docs():
    return [
        Document(page_content="Total assets were 5,118,490.",
                 metadata={"source": "enrg.pdf", "page": 12, "chunk_id": "enrg.pdf::p12::c0"}),
    ]


def test_format_context_tags_source_and_page():
    ctx = format_context(_docs())
    assert "enrg.pdf" in ctx and "12" in ctx
    assert "Total assets were 5,118,490." in ctx


def test_generate_returns_structured_answer():
    answer = RAGAnswer(answer="Total assets were 5,118,490.",
                       citations=[], refused=False, confidence="high")

    class _Structured:
        def invoke(self, _p): return answer

    class _LLM:
        def with_structured_output(self, _schema): return _Structured()

    result = generate("What were total assets?", _docs(), _LLM())
    assert isinstance(result, RAGAnswer)
    assert result.refused is False
    assert "5,118,490" in result.answer


def test_refusal_is_grounded_na_without_llm():
    r = refusal("company not in the provided filings")
    assert r.refused is True
    assert r.answer.strip().upper().startswith("N/A")
    assert r.citations == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_generation.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `backend/rag/generation/schema.py`**

```python
"""Structured answer types for the RAG generator (framework-free — the API
layer maps these to its own response models)."""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str
    company: str = ""
    fiscal_year: str = ""
    page: int = 0


class RAGAnswer(BaseModel):
    answer: str = Field(description="The answer, or 'N/A' if unsupported by the context")
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    confidence: str = Field(default="medium", description="high, medium, or low")
```

- [ ] **Step 4: Implement `backend/rag/generation/generator.py`**

```python
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
```

- [ ] **Step 5: Run to verify pass, then full suite**

Run: `uv run pytest tests/test_generation.py -v` → all pass. Then `uv run pytest -q` → green.

- [ ] **Step 6: Commit**

```bash
git add backend/rag/generation backend/tests/test_generation.py
git commit -m "feat: grounded generation with structured RAGAnswer + refusal"
```

---

### Task 6: Linear query path (basic + hybrid modes)

**Files:**
- Create: `backend/rag/query.py`
- Test: `backend/tests/test_query.py`

**Interfaces:**
- Consumes: everything above + `core.config`, `providers.factory`, `ChromaStore`.
- Produces:
  - `QueryDeps` (dataclass): `store`, `docstore_docs`, `entity_index`, `llm`, `settings` — built once and reused.
  - `build_deps(settings=None) -> QueryDeps` — loads the persisted store, docstore, and `doc_metadata.json` index, and the LLM from the factory.
  - `answer_linear(question: str, mode: str, deps: QueryDeps) -> RAGAnswer` — resolve → (refuse if no sources) → retrieve (`basic`=vector-only, `hybrid`=ensemble) → generate.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_query.py`:
```python
"""Linear query path: refusal short-circuit + retrieve→generate (mocked LLM)."""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.query import QueryDeps, answer_linear
from rag.generation.schema import RAGAnswer
from rag.retrieval.entity_resolver import QueryEntities, build_index
from rag.retrieval.store import ChromaStore


class _HashEmbeddings(Embeddings):
    def _v(self, t):
        v = [0.0] * 8
        for tok in t.lower().split():
            v[hash(tok) % 8] += 1.0
        return v
    def embed_documents(self, texts): return [self._v(t) for t in texts]
    def embed_query(self, text): return self._v(text)


class _LLM:
    """Parses to a fixed entity, and generates a fixed answer."""
    def __init__(self, companies, year):
        self._entities = QueryEntities(companies=companies, fiscal_year=year)
    def with_structured_output(self, schema):
        entities, self_ = self._entities, self
        class _S:
            def invoke(self, _p):
                if schema is QueryEntities:
                    return entities
                return RAGAnswer(answer="Total assets were 5B.", refused=False)
        return _S()


def _deps(tmp_path, companies, year):
    docs = [Document(page_content="cross first total assets five billion",
                     metadata={"source": "cross.pdf", "page": 1, "chunk_id": "cross.pdf::p1::c0"})]
    store = ChromaStore(_HashEmbeddings(), str(tmp_path / "chroma"), "q_test")
    store.add(docs)
    meta = {"cross.pdf": {"company_name": "CrossFirst Bankshares, Inc.",
                          "aliases": ["CrossFirst Bank"], "fiscal_year": "2022"}}

    class _S:  # minimal settings stand-in
        top_k = 5; bm25_weight = 0.4; vector_weight = 0.6

    return QueryDeps(store=store, docstore_docs=docs, entity_index=build_index(meta),
                     llm=_LLM(companies, year), settings=_S())


def test_answer_refuses_wrong_year(tmp_path):
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2023")
    result = answer_linear("assets of CrossFirst Bank in 2023?", "hybrid", deps)
    assert result.refused is True
    assert result.answer.strip().upper().startswith("N/A")


def test_answer_generates_when_resolved(tmp_path):
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2022")
    result = answer_linear("assets of CrossFirst Bank in 2022?", "hybrid", deps)
    assert result.refused is False
    assert "5B" in result.answer
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_query.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `backend/rag/query.py`**

```python
"""Query path: resolve entities -> (refuse | retrieve -> generate)."""

from dataclasses import dataclass

from core.config import Settings, get_settings
from rag.generation.generator import generate, refusal
from rag.generation.schema import RAGAnswer
from rag.providers.factory import get_embeddings, get_llm
from rag.retrieval.docstore import load_docstore
from rag.retrieval.entity_resolver import build_index, parse_query, resolve
from rag.retrieval.hybrid import build_hybrid_retriever, vector_only_retriever
from rag.retrieval.reranker import load_reranker, rerank
from rag.retrieval.store import ChromaStore


@dataclass
class QueryDeps:
    store: ChromaStore
    docstore_docs: list
    entity_index: list
    llm: object
    settings: object
    reranker: object = None  # cross-encoder; None skips reranking (e.g. in tests)


def build_deps(settings: Settings | None = None) -> QueryDeps:
    import json
    from pathlib import Path

    settings = settings or get_settings()
    store = ChromaStore(get_embeddings(settings), settings.chroma_dir, settings.collection_name)
    docstore_docs = load_docstore(settings.docstore_path)
    meta_path = Path(settings.doc_metadata_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return QueryDeps(store=store, docstore_docs=docstore_docs,
                     entity_index=build_index(meta), llm=get_llm(settings),
                     settings=settings, reranker=load_reranker(settings.reranker_model))


def _retrieve(question, mode, sources, deps):
    s = deps.settings
    if mode == "basic":
        retriever = vector_only_retriever(deps.store, sources, s.top_k)
    else:  # hybrid
        retriever = build_hybrid_retriever(deps.store, deps.docstore_docs, sources,
                                           s.top_k, s.bm25_weight, s.vector_weight)
    candidates = retriever.invoke(question)
    if deps.reranker is None:
        return candidates
    return rerank(question, candidates, deps.reranker, s.top_n)


def answer_linear(question: str, mode: str, deps: QueryDeps) -> RAGAnswer:
    entities = parse_query(question, deps.llm)
    res = resolve(entities, deps.entity_index)
    if res.refuse:
        return refusal(f"no filing matches {res.unresolved}")
    docs = _retrieve(question, mode, res.sources, deps)
    if not docs:
        return refusal("no relevant excerpts retrieved")
    return generate(question, docs, deps.llm)
```

- [ ] **Step 4: Run to verify pass, then full suite**

Run: `uv run pytest tests/test_query.py -v` → 2 pass. Then `uv run pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add backend/rag/query.py backend/tests/test_query.py
git commit -m "feat: linear query path (resolve->retrieve->generate) with refusal"
```

---

### Task 7: Agentic query path (LangGraph)

**Files:**
- Create: `backend/rag/agentic.py`
- Test: `backend/tests/test_agentic.py`

**Interfaces:**
- Consumes: `QueryDeps` and the same resolve/retrieve/generate helpers.
- Produces:
  - `AgentState(TypedDict)`: `question: str`, `sources: list[str]`, `unresolved: list[str]`, `docs: list`, `answer: RAGAnswer | None`, `retries: int`.
  - `build_agentic_app(deps)` → a compiled LangGraph: `resolve → route(refuse | retrieve) → grade → route(generate | rewrite→retrieve | refuse)`.
  - `answer_agentic(question: str, deps: QueryDeps) -> RAGAnswer`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_agentic.py`:
```python
"""Agentic path: refusal on unknown entity; answer when resolved (mocked LLM)."""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.agentic import answer_agentic
from rag.generation.schema import RAGAnswer
from rag.query import QueryDeps
from rag.retrieval.entity_resolver import QueryEntities, build_index
from rag.retrieval.store import ChromaStore


class _HashEmbeddings(Embeddings):
    def _v(self, t):
        v = [0.0] * 8
        for tok in t.lower().split():
            v[hash(tok) % 8] += 1.0
        return v
    def embed_documents(self, texts): return [self._v(t) for t in texts]
    def embed_query(self, text): return self._v(text)


class _LLM:
    def __init__(self, companies, year):
        self._e = QueryEntities(companies=companies, fiscal_year=year)
    def with_structured_output(self, schema):
        e = self._e
        class _S:
            def invoke(self, _p):
                if schema is QueryEntities:
                    return e
                return RAGAnswer(answer="Total assets were 5B.", refused=False)
        return _S()


def _deps(tmp_path, companies, year):
    docs = [Document(page_content="cross first total assets five billion",
                     metadata={"source": "cross.pdf", "page": 1, "chunk_id": "cross.pdf::p1::c0"})]
    store = ChromaStore(_HashEmbeddings(), str(tmp_path / "chroma"), "ag_test")
    store.add(docs)
    meta = {"cross.pdf": {"company_name": "CrossFirst Bankshares, Inc.",
                          "aliases": ["CrossFirst Bank"], "fiscal_year": "2022"}}
    class _S:
        top_k = 5; bm25_weight = 0.4; vector_weight = 0.6; max_retries = 1
    return QueryDeps(store=store, docstore_docs=docs, entity_index=build_index(meta),
                     llm=_LLM(companies, year), settings=_S())


def test_agentic_refuses_unknown_company(tmp_path):
    deps = _deps(tmp_path, ["Nonexistent Corp"], "2022")
    result = answer_agentic("assets of Nonexistent Corp in 2022?", deps)
    assert result.refused is True


def test_agentic_answers_when_resolved(tmp_path):
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2022")
    result = answer_agentic("assets of CrossFirst Bank in 2022?", deps)
    assert result.refused is False
    assert "5B" in result.answer
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_agentic.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `backend/rag/agentic.py`**

```python
"""Agentic query path as a LangGraph StateGraph (course: 04_agentic_rag.py).

resolve -> [refuse | retrieve] -> grade -> [generate | rewrite->retrieve | refuse]
The grade step is a simple, transparent check: did retrieval return anything?
If not and retries remain, widen the search once; otherwise refuse.
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
```

- [ ] **Step 4: Run to verify pass, then full suite**

Run: `uv run pytest tests/test_agentic.py -v` → 2 pass. Then `uv run pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add backend/rag/agentic.py backend/tests/test_agentic.py
git commit -m "feat: agentic query path via LangGraph (resolve/retrieve/grade/refuse)"
```

---

### Task 8: Unified `answer(question, mode)` entry point

**Files:**
- Modify: `backend/rag/query.py` (add the dispatcher)
- Test: `backend/tests/test_query.py` (add a dispatch test)

**Interfaces:**
- Produces: `answer(question: str, mode: str, deps: QueryDeps) -> RAGAnswer` — dispatches `basic`/`hybrid` → `answer_linear`, `agentic` → `answer_agentic`. This is the single call Plan 4's API makes.

- [ ] **Step 1: Write the failing test (append to `test_query.py`)**

```python
def test_answer_dispatches_agentic(tmp_path):
    from rag.query import answer
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2022")
    result = answer("assets of CrossFirst Bank in 2022?", "agentic", deps)
    assert result.refused is False
    assert "5B" in result.answer


def test_answer_defaults_to_linear_for_hybrid(tmp_path):
    from rag.query import answer
    deps = _deps(tmp_path, ["CrossFirst Bank"], "2023")
    result = answer("assets of CrossFirst Bank in 2023?", "hybrid", deps)
    assert result.refused is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_query.py -v` → the two new tests FAIL (`answer` not defined).

- [ ] **Step 3: Add the dispatcher to `backend/rag/query.py`**

Append:
```python
def answer(question: str, mode: str, deps: QueryDeps) -> RAGAnswer:
    if mode == "agentic":
        from rag.agentic import answer_agentic
        return answer_agentic(question, deps)
    return answer_linear(question, mode, deps)
```
(The `agentic` import is local to avoid a circular import — `rag.agentic` imports `rag.query`.)

- [ ] **Step 4: Run to verify pass, then full suite**

Run: `uv run pytest tests/test_query.py -v` → all pass. Then `uv run pytest -q` → full suite green.

- [ ] **Step 5: Commit**

```bash
git add backend/rag/query.py backend/tests/test_query.py
git commit -m "feat: unified answer(question, mode) entry point"
```

---

## Self-Review

**Spec coverage (Plan 3 slice):**
- Entity resolver: query parse (`with_structured_output`) + company/year → source, refusal (design §3.3 entity_resolver) → Task 2. ✅
- Hybrid BM25 ⊕ vector filtered by source (design §3.3) → Task 3. ✅
- Simple cross-encoder reranker (top_k → top_n) (design §3.3) → Task 4. ✅
- Refusal-first grounded generation with structured `RAGAnswer` (design §3.4) → Task 5. ✅
- Modes `basic | hybrid | agentic` façade (design §3.3 retriever) → Tasks 6 + 8. ✅
- Agentic LangGraph loop (design §3.3 agentic) → Task 7. ✅
- Score-threshold refusal: simplified to "no docs retrieved → refuse" (a score threshold is a documented later enhancement; the entity resolver already provides the primary refusal).

**Deferred (out of this plan, tracked in `deferred-hardening-backlog.md`):** score-threshold grading in the agentic loop; multi-company `compare` answering beyond retrieving across resolved sources.

**Placeholder scan:** none — full code in every step. Two "verify the import path against the installed version" notes (Tasks 3) describe a concrete contract, not missing content.

**Type consistency:** `QueryEntities`/`Resolution` (Task 2) are consumed by `answer_linear`/agentic (Tasks 6, 7). `RAGAnswer`/`Citation` (Task 5) are the return type everywhere. `QueryDeps` (Task 6, with optional `reranker`) is consumed by Tasks 7, 8. `_retrieve` (Task 6) is reused by Task 7 and applies `rerank` (Task 4) when a reranker is present. `ChromaStore.as_retriever` (Task 3) is the method added in the readability pass. `build_hybrid_retriever`/`vector_only_retriever` signatures (Task 3) match their call sites in `_retrieve`.
