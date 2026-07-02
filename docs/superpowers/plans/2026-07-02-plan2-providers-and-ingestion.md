# Plan 2 — Providers & Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the 20 ERC 10-K PDFs into a persistent, queryable index: a provider abstraction (Groq LLM + cached local HF embeddings), a document-loading + LLM-metadata + chunking pipeline, a persistent Chroma vector store plus a chunk docstore, and a `make ingest` entrypoint that builds it all.

**Architecture:** The framework-free `rag/` core gains three sub-packages — `providers/` (LLM + embeddings factory), `ingestion/` (loaders → metadata → chunking → pipeline), and `retrieval/store.py` (the persistent vector store that ingestion writes to and Plan 3 reads from). Company/fiscal-year/currency metadata is extracted once by the LLM (`with_structured_output`) and cached to a committed `doc_metadata.json`, so re-ingestion is deterministic.

**Tech Stack:** `langchain` (`init_chat_model`, `with_structured_output`), `langchain-groq`, `langchain-huggingface`, `langchain-classic` (`CacheBackedEmbeddings`, `LocalFileStore`), `langchain-chroma` + `chromadb`, `pypdf`, `pdfplumber`, `pydantic` v2, `pytest`.

## Global Constraints

- All environment access goes through `core/config.py` — new settings are ADDED to the existing `Settings` class; no other module reads env vars.
- **UTF-8 everywhere.** The 10-Ks contain non-cp1252 characters (☒, €, £, non-breaking spaces). Never `print()` raw extracted PDF text. Any module with a `__main__`/CLI that may emit document text must call `sys.stdout.reconfigure(encoding="utf-8")` first, and all file reads/writes pass `encoding="utf-8"`.
- **Tests run with NO API key and NO network and NO model download.** Real providers (Groq, HuggingFace) are never instantiated in tests — they are monkeypatched or replaced with in-test fakes. NO repo-wide `conftest.py`.
- **Chroma metadata values must be scalars** (`str`/`int`/`float`/`bool`) — never lists. `aliases` (a list) lives in `doc_metadata.json`, not in chunk metadata.
- PDF extraction: **pypdf for text (primary)**, **pdfplumber for tables (best-effort — may return nothing; never gate on it)**.
- Git commit messages: no AI co-authorship/attribution trailers of any kind.
- Every task ends green (`uv run pytest -q`) and is committed. Work from `backend/`; run tests with `uv run pytest`.

---

### Task 0: Stage the ERC corpus & data settings

**Files:**
- Create: `backend/data/docs/` (20 PDFs copied in)
- Create: `backend/data/benchmark/questions.json`, `backend/data/benchmark/answers.json`
- Create: `backend/data/.gitkeep`
- Modify: repo-root `.gitignore` (keep `data/chroma/`, `data/embeddings_cache/`, `data/docstore.jsonl` ignored; DO commit `data/docs/` and `data/benchmark/`)
- Test: `backend/tests/test_corpus_present.py`

**Interfaces:**
- Consumes: nothing.
- Produces: the on-disk corpus every later task's integration checks reference.

- [ ] **Step 1: Copy the corpus in**

Run (Git Bash):
```bash
cd "/c/Users/asus/Desktop/RagProd/production-rag/backend"
mkdir -p data/docs data/benchmark
cp "/c/Users/asus/Desktop/LangChainRag/data/ERC/pdfs/"*.pdf data/docs/
cp "/c/Users/asus/Desktop/LangChainRag/data/ERC/questions.json" data/benchmark/questions.json
cp "/c/Users/asus/Desktop/LangChainRag/data/ERC/answers.json" data/benchmark/answers.json
ls data/docs/*.pdf | wc -l   # expect 20
```

- [ ] **Step 2: Update repo-root `.gitignore`**

Ensure the RAG-artifacts section reads exactly (replace the existing `backend/data/chroma/` line block):
```
# RAG artifacts (generated — do not commit)
backend/data/chroma/
backend/data/embeddings_cache/
backend/data/docstore.jsonl
backend/data/doc_metadata.json
**/chroma/
```
The PDFs under `backend/data/docs/` and the benchmark JSON under `backend/data/benchmark/` ARE committed (they are the corpus).

- [ ] **Step 3: Write the corpus-presence test**

Create `backend/tests/test_corpus_present.py`:
```python
"""The ERC corpus ships in the repo so the project runs out of the box."""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def test_twenty_pdfs_present():
    pdfs = list((DATA / "docs").glob("*.pdf"))
    assert len(pdfs) == 20


def test_benchmark_files_present_and_aligned():
    questions = json.loads((DATA / "benchmark" / "questions.json").read_text(encoding="utf-8"))
    answers = json.loads((DATA / "benchmark" / "answers.json").read_text(encoding="utf-8"))
    assert len(questions) == len(answers) == 40
    assert questions[0]["question"] == answers[0]["question"]
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_corpus_present.py -v`
Expected: both tests PASS (20 PDFs, 40 aligned Q/A).

- [ ] **Step 5: Commit**

```bash
git add backend/data/docs backend/data/benchmark backend/tests/test_corpus_present.py .gitignore
git commit -m "chore: stage ERC 10-K corpus and benchmark"
```

---

### Task 1: Config extensions & provider factory

**Files:**
- Modify: `backend/core/config.py` (add ingestion/index settings)
- Create: `backend/rag/providers/__init__.py` (empty)
- Create: `backend/rag/providers/base.py`
- Create: `backend/rag/providers/factory.py`
- Test: `backend/tests/test_providers.py`

**Interfaces:**
- Consumes: `core.config.Settings` / `get_settings`.
- Produces:
  - New `Settings` fields (with defaults): `embedding_cache_dir: str = "data/embeddings_cache"`, `doc_metadata_path: str = "data/doc_metadata.json"`, `docstore_path: str = "data/docstore.jsonl"`, `collection_name: str = "financial_reports"`, `metadata_extract_pages: int = 3`, `chunk_size: int = 1000`, `chunk_overlap: int = 150`.
  - `rag/providers/base.py`: type aliases `LLM = BaseChatModel`, `EmbeddingsModel = Embeddings` (langchain's own interfaces ARE the provider abstraction — no reinvented protocol).
  - `rag/providers/factory.py`: `get_llm(settings: Settings | None = None) -> LLM` and `get_embeddings(settings: Settings | None = None) -> EmbeddingsModel`.

- [ ] **Step 1: Add settings fields (edit `core/config.py`)**

In `class Settings`, directly below the existing `data_dir: str = "data/docs"` line, add:
```python
    embedding_cache_dir: str = "data/embeddings_cache"
    doc_metadata_path: str = "data/doc_metadata.json"
    docstore_path: str = "data/docstore.jsonl"
    collection_name: str = "financial_reports"
    metadata_extract_pages: int = 3
    chunk_size: int = 1000
    chunk_overlap: int = 150
```

- [ ] **Step 2: Write the failing test**

Create `backend/tests/test_providers.py`:
```python
"""Provider factory: offline (providers are monkeypatched — no network/model)."""

import pytest
from langchain_core.embeddings import Embeddings

from core.config import Settings
from rag.providers import factory


def _settings(**over) -> Settings:
    base = {"groq_api_key": "test-key", "_env_file": None}
    base.update(over)
    return Settings(**base)


class _CountingEmbeddings(Embeddings):
    """Fake base embeddings that counts embed_documents calls."""

    def __init__(self):
        self.doc_calls = 0

    def embed_documents(self, texts):
        self.doc_calls += 1
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


def test_get_llm_uses_init_chat_model_with_settings(monkeypatch):
    captured = {}

    def fake_init(model, model_provider=None, temperature=None, max_retries=None, **kw):
        captured.update(model=model, provider=model_provider,
                        temperature=temperature, max_retries=max_retries)
        return "FAKE_LLM"

    monkeypatch.setattr(factory, "init_chat_model", fake_init)
    llm = factory.get_llm(_settings(primary_model="llama-3.1-8b-instant",
                                    llm_provider="groq", max_retries=5))
    assert llm == "FAKE_LLM"
    assert captured["model"] == "llama-3.1-8b-instant"
    assert captured["provider"] == "groq"
    assert captured["max_retries"] == 5


def test_get_embeddings_is_cache_backed_and_caches(monkeypatch, tmp_path):
    counting = _CountingEmbeddings()
    monkeypatch.setattr(factory, "_build_base_embeddings", lambda s: counting)

    s = _settings(embedding_cache_dir=str(tmp_path / "emb"),
                  embedding_model="fake-model")
    emb = factory.get_embeddings(s)

    assert isinstance(emb, Embeddings)
    # First call hits the underlying model; second identical call is served from cache.
    emb.embed_documents(["hello world"])
    emb.embed_documents(["hello world"])
    assert counting.doc_calls == 1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_providers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.providers.factory'`.

- [ ] **Step 4: Implement `rag/providers/base.py`**

```python
"""Provider abstraction. LangChain's own interfaces are the abstraction —
swapping providers is a config value, not a code change."""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

LLM = BaseChatModel
EmbeddingsModel = Embeddings

__all__ = ["LLM", "EmbeddingsModel"]
```

- [ ] **Step 5: Implement `rag/providers/factory.py`**

```python
"""Build the configured LLM and embeddings from settings.

Defaults: Groq Llama for generation, local HuggingFace sentence-transformers
for embeddings, wrapped in a filesystem-backed cache so re-ingestion never
recomputes an unchanged chunk's vector.
"""

from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore

from core.config import Settings, get_settings
from rag.providers.base import EmbeddingsModel, LLM


def get_llm(settings: Settings | None = None) -> LLM:
    settings = settings or get_settings()
    return init_chat_model(
        model=settings.primary_model,
        model_provider=settings.llm_provider,
        temperature=0,
        max_retries=settings.max_retries,
    )


def _build_base_embeddings(settings: Settings) -> EmbeddingsModel:
    """The un-cached embedding model. Isolated so tests can swap it."""
    if settings.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.embedding_model)
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model)
    raise ValueError(f"Unknown embedding_provider: {settings.embedding_provider}")


def get_embeddings(settings: Settings | None = None) -> EmbeddingsModel:
    settings = settings or get_settings()
    base = _build_base_embeddings(settings)
    Path(settings.embedding_cache_dir).mkdir(parents=True, exist_ok=True)
    store = LocalFileStore(settings.embedding_cache_dir)
    return CacheBackedEmbeddings.from_bytes_store(
        base, store, namespace=settings.embedding_model
    )
```

- [ ] **Step 6: Run tests to verify they pass, then full suite**

Run: `uv run pytest tests/test_providers.py -v` → both PASS.
Run: `uv run pytest -q` → full suite green (no regressions to config).

- [ ] **Step 7: Commit**

```bash
git add backend/core/config.py backend/rag/providers backend/tests/test_providers.py
git commit -m "feat: add provider factory (Groq LLM + cached HF embeddings)"
```

---

### Task 2: Persistent vector store

**Files:**
- Create: `backend/rag/retrieval/__init__.py` (empty)
- Create: `backend/rag/retrieval/store.py`
- Test: `backend/tests/test_store.py`

**Interfaces:**
- Consumes: `langchain_core.documents.Document`, an `Embeddings` object.
- Produces:
  - `class VectorStore(ABC)` with abstract `add(docs)`, `similarity_search(query, k, filter=None) -> list[Document]`, `similarity_search_with_score(query, k, filter=None) -> list[tuple[Document, float]]`, `count() -> int`.
  - `class ChromaStore(VectorStore)` with `__init__(self, embeddings, persist_directory: str, collection_name: str = "financial_reports")`, implementing the above over `langchain_chroma.Chroma`. Later tasks/plans construct it as `ChromaStore(get_embeddings(), settings.chroma_dir, settings.collection_name)`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_store.py`:
```python
"""ChromaStore over a deterministic fake embedding (no model download, local only)."""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.retrieval.store import ChromaStore


class _HashEmbeddings(Embeddings):
    """Deterministic 8-dim embedding from token hashes — no network."""

    def _vec(self, text: str):
        v = [0.0] * 8
        for tok in text.lower().split():
            v[hash(tok) % 8] += 1.0
        return v

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def _docs():
    return [
        Document(page_content="total assets were five billion dollars",
                 metadata={"company": "TransUnion", "fiscal_year": "2022"}),
        Document(page_content="net income was eighty eight million",
                 metadata={"company": "Petra Diamonds", "fiscal_year": "2022"}),
    ]


def test_add_and_count(tmp_path):
    store = ChromaStore(_HashEmbeddings(), str(tmp_path / "chroma"), "t_count")
    store.add(_docs())
    assert store.count() == 2


def test_similarity_search_returns_documents(tmp_path):
    store = ChromaStore(_HashEmbeddings(), str(tmp_path / "chroma"), "t_search")
    store.add(_docs())
    results = store.similarity_search("total assets", k=1)
    assert len(results) == 1
    assert isinstance(results[0], Document)


def test_similarity_search_with_score_returns_pairs(tmp_path):
    store = ChromaStore(_HashEmbeddings(), str(tmp_path / "chroma"), "t_score")
    store.add(_docs())
    pairs = store.similarity_search_with_score("net income", k=2)
    assert len(pairs) == 2
    doc, score = pairs[0]
    assert isinstance(doc, Document)
    assert isinstance(score, float)


def test_metadata_filter_restricts_results(tmp_path):
    store = ChromaStore(_HashEmbeddings(), str(tmp_path / "chroma"), "t_filter")
    store.add(_docs())
    results = store.similarity_search("anything", k=5, filter={"company": "Petra Diamonds"})
    assert results
    assert all(d.metadata["company"] == "Petra Diamonds" for d in results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.retrieval.store'`.

- [ ] **Step 3: Implement `rag/retrieval/store.py`**

```python
"""Persistent vector store behind a small interface.

ChromaStore is the default (local, zero-infra). The VectorStore ABC is the
seam a pgvector/Qdrant adapter would implement for a shared, scaled index.
"""

from abc import ABC, abstractmethod

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStore(ABC):
    @abstractmethod
    def add(self, docs: list[Document]) -> None: ...

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 5, filter: dict | None = None
    ) -> list[Document]: ...

    @abstractmethod
    def similarity_search_with_score(
        self, query: str, k: int = 5, filter: dict | None = None
    ) -> list[tuple[Document, float]]: ...

    @abstractmethod
    def count(self) -> int: ...


class ChromaStore(VectorStore):
    def __init__(
        self,
        embeddings: Embeddings,
        persist_directory: str,
        collection_name: str = "financial_reports",
    ):
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )

    def add(self, docs: list[Document]) -> None:
        if not docs:
            return
        # Use deterministic chunk_id as the vector id so re-ingesting the same
        # corpus upserts (no duplicates) rather than appending.
        ids = [d.metadata["chunk_id"] for d in docs] if all(
            "chunk_id" in d.metadata for d in docs
        ) else None
        self._store.add_documents(docs, ids=ids)

    def similarity_search(
        self, query: str, k: int = 5, filter: dict | None = None
    ) -> list[Document]:
        return self._store.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self, query: str, k: int = 5, filter: dict | None = None
    ) -> list[tuple[Document, float]]:
        return self._store.similarity_search_with_score(query, k=k, filter=filter)

    def count(self) -> int:
        return self._store._collection.count()
```

- [ ] **Step 4: Run test to verify it passes, then full suite**

Run: `uv run pytest tests/test_store.py -v` → 4 PASS.
Run: `uv run pytest -q` → green.

Note: if `similarity_search_with_score`'s filter arg name differs in the installed `langchain-chroma`, keep the public method signatures above unchanged and adapt only the internal call; the tests define the contract.

- [ ] **Step 5: Commit**

```bash
git add backend/rag/retrieval backend/tests/test_store.py
git commit -m "feat: add persistent Chroma vector store behind an interface"
```

---

### Task 3: Document loaders

**Files:**
- Create: `backend/rag/ingestion/__init__.py` (empty)
- Create: `backend/rag/ingestion/loaders.py`
- Test: `backend/tests/test_loaders.py`

**Interfaces:**
- Consumes: `pypdf`, `pdfplumber`, `langchain_core.documents.Document`.
- Produces:
  - `serialize_tables(tables: list[list[list]]) -> str` — turn pdfplumber tables into pipe-delimited text (empty string if none).
  - `load_pdf(path: str | Path) -> list[Document]` — one Document per page, `page_content` = page text (pypdf) + serialized tables (pdfplumber, best-effort), metadata `{"source": <filename>, "page": <1-based>}`.
  - `load_text(path) -> list[Document]` — single Document for `.md`/`.txt`, metadata `{"source": <filename>, "page": 1}`.
  - `load_document(path) -> list[Document]` — dispatch by extension (`.pdf` → `load_pdf`, `.md`/`.txt` → `load_text`, else `ValueError`).
  - `load_directory(dir_path) -> Iterator[list[Document]]` — lazily yield per-file document lists for every supported file in the dir (sorted).

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_loaders.py`:
```python
"""Loader logic tested offline: text/markdown via temp files, PDF via fakes."""

from pathlib import Path

from langchain_core.documents import Document

from rag.ingestion import loaders


def test_serialize_tables_joins_rows_and_cells():
    tables = [[["Assets", "2022"], ["Total", "5,118,490"]]]
    out = loaders.serialize_tables(tables)
    assert "Assets | 2022" in out
    assert "Total | 5,118,490" in out


def test_serialize_tables_empty_returns_empty_string():
    assert loaders.serialize_tables([]) == ""
    assert loaders.serialize_tables([[]]) == ""


def test_load_text_reads_markdown(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# Heading\n\nBody text.", encoding="utf-8")
    docs = loaders.load_text(p)
    assert len(docs) == 1
    assert docs[0].metadata == {"source": "note.md", "page": 1}
    assert "Body text." in docs[0].page_content


def test_load_document_dispatches_and_rejects_unknown(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b", encoding="utf-8")
    import pytest
    with pytest.raises(ValueError):
        loaders.load_document(p)


def test_load_pdf_merges_text_and_tables_with_page_metadata(monkeypatch, tmp_path):
    # Fake pypdf: two pages of text.
    class _Page:
        def __init__(self, t): self._t = t
        def extract_text(self): return self._t

    class _Reader:
        def __init__(self, path): self.pages = [_Page("Net income 88"), _Page("Total assets 5,118,490")]

    monkeypatch.setattr(loaders.pypdf, "PdfReader", _Reader)

    # Fake pdfplumber: page 1 has a table, page 2 has none.
    class _PlPage:
        def __init__(self, tables): self._tables = tables
        def extract_tables(self): return self._tables

    class _Pdf:
        def __init__(self): self.pages = [_PlPage([[["Rev", "10"]]]), _PlPage([])]
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(loaders.pdfplumber, "open", lambda path: _Pdf())

    pdf = tmp_path / "84749ef5.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")
    docs = loaders.load_pdf(pdf)

    assert len(docs) == 2
    assert docs[0].metadata == {"source": "84749ef5.pdf", "page": 1}
    assert "Net income 88" in docs[0].page_content
    assert "Rev | 10" in docs[0].page_content        # table merged on page 1
    assert docs[1].metadata["page"] == 2
    assert "Total assets 5,118,490" in docs[1].page_content


def test_load_directory_yields_supported_files(tmp_path):
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("bravo", encoding="utf-8")
    (tmp_path / "skip.csv").write_text("nope", encoding="utf-8")
    batches = list(loaders.load_directory(tmp_path))
    sources = sorted(d.metadata["source"] for batch in batches for d in batch)
    assert sources == ["a.md", "b.txt"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_loaders.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.ingestion.loaders'`.

- [ ] **Step 3: Implement `rag/ingestion/loaders.py`**

```python
"""Load PDFs (pypdf text + pdfplumber tables best-effort) and text files.

pypdf reliably extracts the narrative text and inline financial figures.
pdfplumber table detection is line-based and often finds nothing in
whitespace-aligned financial statements — it is a best-effort bonus, never
a requirement (the numbers are already in the pypdf text).
"""

from collections.abc import Iterator
from pathlib import Path

import pdfplumber
import pypdf
from langchain_core.documents import Document

_TEXT_EXT = {".md", ".txt"}


def serialize_tables(tables: list) -> str:
    lines: list[str] = []
    for table in tables or []:
        for row in table or []:
            cells = [("" if c is None else str(c).strip()) for c in row]
            if any(cells):
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def load_pdf(path: str | Path) -> list[Document]:
    path = Path(path)
    reader = pypdf.PdfReader(str(path))
    # Best-effort table extraction; tolerate any pdfplumber failure.
    tables_by_page: dict[int, str] = {}
    try:
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                serialized = serialize_tables(page.extract_tables())
                if serialized:
                    tables_by_page[i] = serialized
    except Exception:
        tables_by_page = {}

    docs: list[Document] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        table_text = tables_by_page.get(i, "")
        content = text if not table_text else f"{text}\n\n{table_text}"
        docs.append(
            Document(page_content=content, metadata={"source": path.name, "page": i + 1})
        )
    return docs


def load_text(path: str | Path) -> list[Document]:
    path = Path(path)
    content = path.read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=content, metadata={"source": path.name, "page": 1})]


def load_document(path: str | Path) -> list[Document]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext in _TEXT_EXT:
        return load_text(path)
    raise ValueError(f"Unsupported file type: {ext}")


def load_directory(dir_path: str | Path) -> Iterator[list[Document]]:
    dir_path = Path(dir_path)
    supported = {".pdf", *_TEXT_EXT}
    for p in sorted(dir_path.iterdir()):
        if p.is_file() and p.suffix.lower() in supported:
            yield load_document(p)
```

- [ ] **Step 4: Run test to verify it passes, then full suite**

Run: `uv run pytest tests/test_loaders.py -v` → 6 PASS.
Run: `uv run pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add backend/rag/ingestion/__init__.py backend/rag/ingestion/loaders.py backend/tests/test_loaders.py
git commit -m "feat: add PDF/text loaders (pypdf text + pdfplumber tables)"
```

---

### Task 4: LLM metadata extraction (cached)

**Files:**
- Create: `backend/rag/ingestion/metadata.py`
- Test: `backend/tests/test_metadata.py`

**Interfaces:**
- Consumes: an `LLM` (from `providers.factory.get_llm`), `pydantic`.
- Produces:
  - `class DocumentMetadata(BaseModel)`: `company_name: str`, `aliases: list[str] = []`, `ticker: str | None = None`, `fiscal_year: str`, `period_end_date: str | None = None`, `reporting_currency: str | None = None`, `report_type: str | None = None`.
  - `extract_metadata(doc_text: str, llm) -> DocumentMetadata` — calls `llm.with_structured_output(DocumentMetadata).invoke(prompt)`.
  - `first_pages_text(pages: list[Document], n: int) -> str` — concatenate the first `n` pages' content.
  - `load_or_extract(source_name: str, pages: list[Document], llm, cache_path: str, n_pages: int) -> DocumentMetadata` — return cached metadata for `source_name` from the JSON cache if present, else extract via LLM, write back to the cache, and return it. Cache file maps `{source_name: DocumentMetadata-dict}`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_metadata.py`:
```python
"""Metadata extraction + JSON caching, tested with a fake LLM (no network)."""

import json

from langchain_core.documents import Document

from rag.ingestion import metadata as M
from rag.ingestion.metadata import DocumentMetadata


class _FakeStructuredLLM:
    def __init__(self, result): self._result = result
    def invoke(self, _prompt): return self._result


class _FakeLLM:
    """Records how many times structured extraction is invoked."""

    def __init__(self, result):
        self._result = result
        self.calls = 0

    def with_structured_output(self, _schema):
        self.calls += 1
        return _FakeStructuredLLM(self._result)


def _md():
    return DocumentMetadata(company_name="TransUnion", aliases=["TransUnion LLC"],
                            ticker="TRU", fiscal_year="2022",
                            period_end_date="2022-12-31", reporting_currency="USD",
                            report_type="10-K")


def test_first_pages_text_concatenates_n_pages():
    pages = [Document(page_content=f"page {i}", metadata={"page": i}) for i in range(1, 6)]
    text = M.first_pages_text(pages, 3)
    assert "page 1" in text and "page 3" in text and "page 4" not in text


def test_extract_metadata_calls_structured_output():
    llm = _FakeLLM(_md())
    md = M.extract_metadata("TransUnion 2022 annual report ...", llm)
    assert md.company_name == "TransUnion"
    assert md.reporting_currency == "USD"
    assert llm.calls == 1


def test_load_or_extract_writes_then_reads_cache(tmp_path):
    cache = tmp_path / "doc_metadata.json"
    pages = [Document(page_content="TransUnion 2022", metadata={"page": 1})]
    llm = _FakeLLM(_md())

    first = M.load_or_extract("tru.pdf", pages, llm, str(cache), n_pages=3)
    assert first.company_name == "TransUnion"
    assert llm.calls == 1
    assert cache.exists()
    stored = json.loads(cache.read_text(encoding="utf-8"))
    assert stored["tru.pdf"]["reporting_currency"] == "USD"

    # Second call for the same source uses the cache — LLM not invoked again.
    second = M.load_or_extract("tru.pdf", pages, llm, str(cache), n_pages=3)
    assert second.company_name == "TransUnion"
    assert llm.calls == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metadata.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.ingestion.metadata'`.

- [ ] **Step 3: Implement `rag/ingestion/metadata.py`**

```python
"""Extract reporting-entity metadata from a filing's first pages via the LLM,
cached to a JSON file so re-ingestion is deterministic and reviewable."""

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured metadata from the opening pages of a company "
            "annual report or SEC filing. Identify the SINGLE reporting entity (the "
            "company the report is FOR, not its auditor, exchange, or subsidiaries). "
            "Use the fiscal year the financial statements cover. If a field is not "
            "stated, leave it null. Currency is the reporting currency of the primary "
            "financial statements (ISO code like USD, EUR, GBP, JPY, CHF).",
        ),
        ("human", "Opening pages:\n\n{text}"),
    ]
)


class DocumentMetadata(BaseModel):
    company_name: str = Field(description="Reporting entity's name")
    aliases: list[str] = Field(default_factory=list,
                               description="Other names/legal forms for the same entity")
    ticker: str | None = Field(default=None, description="Stock ticker if stated")
    fiscal_year: str = Field(description="Fiscal year the statements cover, e.g. '2022'")
    period_end_date: str | None = Field(default=None,
                                        description="Fiscal period end, ISO date if known")
    reporting_currency: str | None = Field(default=None, description="ISO currency code")
    report_type: str | None = Field(default=None, description="e.g. '10-K', 'Annual Report'")


def first_pages_text(pages: list[Document], n: int) -> str:
    return "\n\n".join(p.page_content for p in pages[:n])


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
) -> DocumentMetadata:
    path = Path(cache_path)
    cache = _read_cache(path)
    if source_name in cache:
        return DocumentMetadata(**cache[source_name])

    md = extract_metadata(first_pages_text(pages, n_pages), llm)
    cache[source_name] = md.model_dump()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return md
```

- [ ] **Step 4: Run test to verify it passes, then full suite**

Run: `uv run pytest tests/test_metadata.py -v` → 3 PASS.
Run: `uv run pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add backend/rag/ingestion/metadata.py backend/tests/test_metadata.py
git commit -m "feat: add cached LLM metadata extraction for filings"
```

---

### Task 5: Chunking with metadata

**Files:**
- Create: `backend/rag/ingestion/chunking.py`
- Test: `backend/tests/test_chunking.py`

**Interfaces:**
- Consumes: `langchain_text_splitters.RecursiveCharacterTextSplitter`, `DocumentMetadata`.
- Produces:
  - `chunk_pages(pages: list[Document], metadata: DocumentMetadata, chunk_size: int, chunk_overlap: int) -> list[Document]` — split each page, and stamp every resulting chunk's metadata with SCALAR fields (`source`, `page` from the page, plus `company`, `fiscal_year`, `period_end_date`, `reporting_currency`, `ticker`, `report_type` from `metadata`), plus a `chunk_id` (`"<source>::p<page>::c<index>"`). `aliases` is a list → NOT added to chunk metadata (Chroma rejects lists).

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_chunking.py`:
```python
"""Chunking preserves page + entity metadata as Chroma-safe scalars."""

from langchain_core.documents import Document

from rag.ingestion.chunking import chunk_pages
from rag.ingestion.metadata import DocumentMetadata


def _md():
    return DocumentMetadata(company_name="Petra Diamonds", aliases=["Petra Diamonds Ltd"],
                            ticker="PDL", fiscal_year="2022",
                            period_end_date="2022-06-30", reporting_currency="GBP",
                            report_type="Annual Report")


def _pages():
    long_text = "Financial statements. " * 80  # forces multiple chunks
    return [
        Document(page_content=long_text, metadata={"source": "petra.pdf", "page": 3}),
        Document(page_content="Short note.", metadata={"source": "petra.pdf", "page": 4}),
    ]


def test_chunks_carry_entity_and_page_metadata():
    chunks = chunk_pages(_pages(), _md(), chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 2  # page 3 split into several
    c = chunks[0]
    assert c.metadata["company"] == "Petra Diamonds"
    assert c.metadata["fiscal_year"] == "2022"
    assert c.metadata["reporting_currency"] == "GBP"
    assert c.metadata["source"] == "petra.pdf"
    assert c.metadata["page"] == 3


def test_chunk_metadata_has_no_list_values():
    chunks = chunk_pages(_pages(), _md(), chunk_size=200, chunk_overlap=20)
    for c in chunks:
        assert "aliases" not in c.metadata
        assert all(not isinstance(v, list) for v in c.metadata.values())


def test_chunk_ids_are_unique_and_traceable():
    chunks = chunk_pages(_pages(), _md(), chunk_size=200, chunk_overlap=20)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0].startswith("petra.pdf::p3::c")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.ingestion.chunking'`.

- [ ] **Step 3: Implement `rag/ingestion/chunking.py`**

```python
"""Split pages into chunks, stamping each with Chroma-safe scalar metadata."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.ingestion.metadata import DocumentMetadata


def _entity_fields(md: DocumentMetadata) -> dict:
    """Scalar-only entity metadata (no lists — Chroma rejects them)."""
    return {
        "company": md.company_name,
        "fiscal_year": md.fiscal_year,
        "period_end_date": md.period_end_date or "",
        "reporting_currency": md.reporting_currency or "",
        "ticker": md.ticker or "",
        "report_type": md.report_type or "",
    }


def chunk_pages(
    pages: list[Document],
    metadata: DocumentMetadata,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    entity = _entity_fields(metadata)
    out: list[Document] = []
    for page in pages:
        source = page.metadata.get("source", "unknown")
        page_no = page.metadata.get("page", 0)
        for idx, piece in enumerate(splitter.split_text(page.page_content)):
            if not piece.strip():
                continue
            meta = {
                "source": source,
                "page": page_no,
                "chunk_id": f"{source}::p{page_no}::c{idx}",
                **entity,
            }
            out.append(Document(page_content=piece, metadata=meta))
    return out
```

- [ ] **Step 4: Run test to verify it passes, then full suite**

Run: `uv run pytest tests/test_chunking.py -v` → 3 PASS.
Run: `uv run pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add backend/rag/ingestion/chunking.py backend/tests/test_chunking.py
git commit -m "feat: add metadata-preserving chunking"
```

---

### Task 6: Ingestion pipeline & `make ingest`

**Files:**
- Create: `backend/rag/ingestion/pipeline.py`
- Create: `backend/Makefile`
- Test: `backend/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `loaders.load_directory`, `metadata.load_or_extract`, `chunking.chunk_pages`, `retrieval.store.ChromaStore`, `providers.factory` (`get_llm`, `get_embeddings`), `core.config.get_settings`.
- Produces:
  - `ingest(settings, store, llm) -> dict` — for each file in `settings.data_dir`: load pages → `load_or_extract` metadata → `chunk_pages` → `store.add(chunks)` → append chunks to the docstore JSONL at `settings.docstore_path`. Returns a summary dict `{"files": int, "chunks": int, "companies": list[str]}`. Idempotent-friendly: rewrites the docstore fresh each run.
  - `main() -> None` — CLI entrypoint: force UTF-8 stdout, build real `store`+`llm` from settings, run `ingest`, print the summary (summary only — never raw document text).

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_pipeline.py`:
```python
"""Pipeline wiring tested with fakes: fake embeddings/LLM, temp dirs, tiny docs."""

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from core.config import Settings
from rag.ingestion import pipeline
from rag.ingestion.metadata import DocumentMetadata
from rag.retrieval.store import ChromaStore


class _HashEmbeddings(Embeddings):
    def _vec(self, t):
        v = [0.0] * 8
        for tok in t.lower().split():
            v[hash(tok) % 8] += 1.0
        return v
    def embed_documents(self, texts): return [self._vec(t) for t in texts]
    def embed_query(self, text): return self._vec(text)


class _FakeStructured:
    def __init__(self, name): self._name = name
    def invoke(self, _p):
        return DocumentMetadata(company_name=self._name, fiscal_year="2022",
                                reporting_currency="USD", report_type="10-K")


class _FakeLLM:
    def with_structured_output(self, _schema):
        # Company name derived from nothing here; fixed for determinism.
        return _FakeStructured("TestCorp")


def _settings(tmp_path) -> Settings:
    return Settings(
        groq_api_key="test-key", _env_file=None,
        data_dir=str(tmp_path / "docs"),
        chroma_dir=str(tmp_path / "chroma"),
        docstore_path=str(tmp_path / "docstore.jsonl"),
        doc_metadata_path=str(tmp_path / "doc_metadata.json"),
        chunk_size=200, chunk_overlap=20, metadata_extract_pages=2,
    )


def test_ingest_populates_store_and_docstore(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("Total assets " * 60, encoding="utf-8")
    (docs_dir / "b.md").write_text("Net income " * 60, encoding="utf-8")

    settings = _settings(tmp_path)
    store = ChromaStore(_HashEmbeddings(), settings.chroma_dir, settings.collection_name)

    summary = pipeline.ingest(settings, store, _FakeLLM())

    assert summary["files"] == 2
    assert summary["chunks"] > 2
    assert store.count() == summary["chunks"]

    # Docstore JSONL has one line per chunk, each with entity metadata.
    lines = Path(settings.docstore_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == summary["chunks"]
    first = json.loads(lines[0])
    assert first["metadata"]["company"] == "TestCorp"
    assert "page_content" in first

    # doc_metadata.json cached for both sources.
    md_cache = json.loads(Path(settings.doc_metadata_path).read_text(encoding="utf-8"))
    assert set(md_cache.keys()) == {"a.txt", "b.md"}


def test_reingest_is_idempotent(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("Total assets " * 60, encoding="utf-8")
    settings = _settings(tmp_path)
    store = ChromaStore(_HashEmbeddings(), settings.chroma_dir, settings.collection_name)

    s1 = pipeline.ingest(settings, store, _FakeLLM())
    count_after_first = store.count()
    lines_after_first = Path(settings.docstore_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines_after_first) == s1["chunks"] == count_after_first

    # Re-ingesting the same corpus must NOT duplicate vectors (deterministic
    # chunk_id upserts) and must rewrite the docstore fresh (not append).
    s2 = pipeline.ingest(settings, store, _FakeLLM())
    assert store.count() == count_after_first
    lines_after_second = Path(settings.docstore_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines_after_second) == s2["chunks"] == count_after_first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.ingestion.pipeline'`.

- [ ] **Step 3: Implement `rag/ingestion/pipeline.py`**

```python
"""Batch ingestion: load → metadata → chunk → embed/persist + docstore.

Run offline via `python -m rag.ingestion.pipeline` (see `make ingest`).
"""

import json
import sys
from pathlib import Path

from core.config import Settings, get_settings
from rag.ingestion.chunking import chunk_pages
from rag.ingestion.loaders import load_directory
from rag.ingestion.metadata import load_or_extract
from rag.providers.factory import get_embeddings, get_llm
from rag.retrieval.store import ChromaStore, VectorStore


def ingest(settings: Settings, store: VectorStore, llm) -> dict:
    docstore = Path(settings.docstore_path)
    docstore.parent.mkdir(parents=True, exist_ok=True)

    files = 0
    total_chunks = 0
    companies: list[str] = []

    # Fresh docstore each run so it stays in lock-step with the vector store.
    with docstore.open("w", encoding="utf-8") as fh:
        for pages in load_directory(settings.data_dir):
            if not pages:
                continue
            source = pages[0].metadata.get("source", "unknown")
            md = load_or_extract(
                source, pages, llm, settings.doc_metadata_path,
                settings.metadata_extract_pages,
            )
            chunks = chunk_pages(pages, md, settings.chunk_size, settings.chunk_overlap)
            store.add(chunks)
            for c in chunks:
                fh.write(json.dumps(
                    {"page_content": c.page_content, "metadata": c.metadata},
                    ensure_ascii=False,
                ) + "\n")
            files += 1
            total_chunks += len(chunks)
            companies.append(md.company_name)

    return {"files": files, "chunks": total_chunks, "companies": companies}


def main() -> None:
    # The 10-Ks contain non-cp1252 characters; force UTF-8 before any output.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    settings = get_settings()
    store = ChromaStore(get_embeddings(settings), settings.chroma_dir, settings.collection_name)
    summary = ingest(settings, store, get_llm(settings))
    print(f"Ingested {summary['files']} files -> {summary['chunks']} chunks")
    print(f"Companies: {', '.join(sorted(set(summary['companies'])))}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `backend/Makefile`**

```make
.PHONY: ingest run ui eval test

ingest:
	PYTHONUTF8=1 uv run python -m rag.ingestion.pipeline

test:
	uv run pytest -q

run:
	@echo "API server arrives in a later plan"

eval:
	@echo "Eval harness arrives in a later plan"
```

- [ ] **Step 5: Run test to verify it passes, then full suite**

Run: `uv run pytest tests/test_pipeline.py -v` → 2 PASS.
Run: `uv run pytest -q` → full suite green.

- [ ] **Step 6: Commit**

```bash
git add backend/rag/ingestion/pipeline.py backend/Makefile backend/tests/test_pipeline.py
git commit -m "feat: add ingestion pipeline and make ingest entrypoint"
```

- [ ] **Step 7: Real ingest smoke run (manual, needs GROQ_API_KEY + network + model download)**

This step is NOT a test — it validates the real pipeline end-to-end and is run once by the controller (it downloads the HF embedding model and makes ~20 Groq metadata calls). Only run if a real `GROQ_API_KEY` is in `backend/.env`:
```bash
cd backend && PYTHONUTF8=1 uv run python -m rag.ingestion.pipeline
```
Expected: prints "Ingested 20 files -> <N> chunks" and a company list; creates `data/chroma/`, `data/docstore.jsonl`, `data/doc_metadata.json`. Review `data/doc_metadata.json` for extraction quality (company names + fiscal years look right). Do NOT commit the generated artifacts (they are gitignored); DO commit `data/doc_metadata.json` only if you want the extracted mapping versioned — decide during the run.

---

## Self-Review

**Spec coverage (Plan 2 slice of the design):**
- Provider abstraction + Groq/HF defaults + `init_chat_model` + `CacheBackedEmbeddings` (design §3.1) → Task 1. ✅
- Persistent Chroma behind `VectorStore` interface with `similarity_search_with_score` + metadata filtering (design §3.3 store) → Task 2. ✅
- Loaders: pypdf text + pdfplumber tables best-effort, DirectoryLoader/lazy, page metadata (design §3.2) → Task 3. ✅
- LLM structured metadata extraction (company/aliases/fiscal_year/period_end/currency), cached to `doc_metadata.json` (design §3.2 + user decision) → Task 4. ✅
- Metadata-preserving, Chroma-safe chunking (design §3.2) → Task 5. ✅
- Ingestion pipeline + docstore + `make ingest`, UTF-8 safe (design §3.2, §4.1, §7) → Task 6. ✅
- Corpus staged in-repo (design §7 runnability) → Task 0. ✅
- Deferred to Plan 3: hybrid/BM25 retriever (built from the docstore), reranker, agentic loop, generation. Deferred to Plan 4: API. The `similarity_search` filter/score surface is ready for them.

**Placeholder scan:** none — full code in every step. Two explicit "adapt only if the installed library API differs" notes (Task 2 Step 4, and the real-ingest step) describe verification against a concrete contract, not missing content.

**Type consistency:** `DocumentMetadata` fields defined in Task 4 are consumed by `chunk_pages` in Task 5 (`company_name`, `fiscal_year`, `period_end_date`, `reporting_currency`, `ticker`, `report_type`) and by `load_or_extract` in Task 6. `ChromaStore(embeddings, persist_directory, collection_name)` signature (Task 2) matches its construction in Task 6's `main()` and in `test_pipeline`. `ingest(settings, store, llm)` signature (Task 6) matches its test. Settings fields added in Task 1 (`docstore_path`, `doc_metadata_path`, `collection_name`, `metadata_extract_pages`, `chunk_size`, `chunk_overlap`, `embedding_cache_dir`) are the ones every later task reads.
