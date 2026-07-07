# Plan 5 — Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score the system against the 40-question ERC benchmark — run every question through `answer(question, mode)` for each retrieval mode, match predictions to the golden answers (with unit/scale-tolerant numeric matching and `N/A` handling), and produce a **per-category × per-mode** results table plus a committed `eval_results.json` the dashboard will read.

**Architecture:** A small, framework-free `eval/` package that consumes `rag.query.answer`. The pieces are independently testable: `golden` (load the benchmark), `matching` (the correctness verdict — the crux), `runner` (run a mode over the set), `metrics` (aggregate per category/mode), `evaluate` (CLI orchestrator). No LLM or network in tests — the runner takes an injectable `answer_fn`.

**Tech Stack:** stdlib + `pydantic` (already present). The real run uses `rag.query.build_deps` + `answer` (Groq + the built index), but that path is exercised only by the CLI, never by tests.

## Global Constraints

- `eval/` consumes `rag/` (it's a layer above); `rag/` must not import `eval/`.
- Tests run with NO API key / network / model — the runner is tested with a fake `answer_fn`; matching/golden/metrics are pure. NO repo-wide `conftest.py`.
- The golden truth is `data/benchmark/answers.json` (it has `question`, `schema`, `answer` list, `category`); `questions.json` is only the input list. Read both as UTF-8.
- A prediction is correct if it matches **any** element of the golden `answer` list (the list is the set of acceptable answers, sometimes mixing a value and `"N/A"`).
- Keep code direct and readable; standard library idioms; commit messages carry no AI-attribution trailers.
- Every task ends green (`uv run pytest -q`) and is committed. Work from `backend/`.

---

### Task 1: Golden benchmark loader

**Files:**
- Create: `backend/eval/golden.py`
- Test: `backend/tests/test_golden.py`

**Interfaces:**
- Produces:
  - `GoldenItem(BaseModel)`: `question: str`, `answer_type: str` (`"number"` | `"name"`), `answers: list` (values may be numbers or strings incl. `"N/A"`), `category: str` (default `"other"`). (Field is `answer_type`, not `schema`, to avoid shadowing Pydantic's `BaseModel.schema`.)
  - `load_golden(path: str) -> list[GoldenItem]` — parse `answers.json` (maps its `answer` field → `answers`, missing `category` → `"other"`).

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_golden.py`:
```python
"""Load the golden benchmark (answers.json) into typed items."""

import json

from eval.golden import GoldenItem, load_golden


def test_load_golden_parses_items(tmp_path):
    p = tmp_path / "answers.json"
    p.write_text(json.dumps([
        {"question": "net income of Petra 2022?", "schema": "number",
         "answer": [88100000], "comment": "x", "category": "retrieval"},
        {"question": "liabilities of CrossFirst 2023?", "schema": "number",
         "answer": ["N/A"], "comment": "wrong year", "category": "hallucination"},
        {"question": "no category here", "schema": "name", "answer": ["MITSUI"]},
    ]), encoding="utf-8")

    items = load_golden(str(p))
    assert len(items) == 3
    assert isinstance(items[0], GoldenItem)
    assert items[0].answers == [88100000]
    assert items[0].category == "retrieval"
    assert items[2].category == "other"  # missing category defaults
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_golden.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `backend/eval/golden.py`**

```python
"""Load the ERC golden benchmark (answers.json) into typed items."""

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldenItem(BaseModel):
    question: str
    answer_type: str  # "number" | "name"  (not `schema`: shadows BaseModel.schema)
    answers: list = Field(default_factory=list)  # acceptable answers (values or "N/A")
    category: str = "other"


def load_golden(path: str) -> list[GoldenItem]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        GoldenItem(
            question=row["question"],
            answer_type=row.get("schema", "number"),
            answers=row.get("answer", []),
            category=row.get("category", "other"),
        )
        for row in rows
    ]
```

- [ ] **Step 4: Run to verify pass, then full suite**

Run: `uv run pytest tests/test_golden.py -v` → pass. Then `uv run pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add backend/eval/golden.py backend/tests/test_golden.py
git commit -m "feat(eval): load golden benchmark into typed items"
```

---

### Task 2: Answer matching (the correctness verdict)

**Files:**
- Create: `backend/eval/matching.py`
- Test: `backend/tests/test_matching.py`

**Interfaces:**
- Consumes: `RAGAnswer` (from `rag.generation.schema`), `GoldenItem`.
- Produces:
  - `parse_number(text: str) -> float | None` — first number in text; handles commas, decimals, parentheses-negatives, `%`, currency symbols.
  - `number_matches(pred: float, golden: float, rel_tol=0.02) -> bool` — true if `pred` equals `golden` within relative tolerance at **any scale** (`×1, 1e3, 1e6, 1e9` and inverses) so `88.1` (millions) matches `88100000`.
  - `name_matches(pred: str, golden: str) -> bool` — normalized-token subset match (`"MITSUI O.S.K. Lines"` ≈ `"MITSUI O.S.K. LINES"`).
  - `is_correct(prediction: RAGAnswer, item: GoldenItem) -> bool` — a refusal is correct iff `"N/A"` is acceptable; otherwise the answer must match some numeric/name golden value.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_matching.py`:
```python
"""Answer matching: numeric scale tolerance, N/A, names, multi-valid."""

from eval.golden import GoldenItem
from eval.matching import is_correct, name_matches, number_matches, parse_number
from rag.generation.schema import RAGAnswer


def test_parse_number_handles_formats():
    assert parse_number("88.1") == 88.1
    assert parse_number("88,100,000") == 88100000
    assert parse_number("(4,432)") == -4432
    assert parse_number("15.63%") == 15.63
    assert parse_number("no number here") is None


def test_number_matches_across_scales():
    assert number_matches(88.1, 88100000)       # millions vs absolute
    assert number_matches(88100000, 88.1)       # symmetric
    assert not number_matches(88.1, 999)


def test_name_matches_normalizes():
    assert name_matches("MITSUI O.S.K. Lines", "MITSUI O.S.K. LINES")
    assert not name_matches("TransUnion", "Petra Diamonds")


def _ans(answer="", refused=False):
    return RAGAnswer(answer=answer, refused=refused)


def test_refusal_correct_when_na_acceptable():
    item = GoldenItem(question="q", answer_type="number", answers=["N/A"], category="hallucination")
    assert is_correct(_ans(answer="N/A", refused=True), item) is True
    # answering a number when only N/A is acceptable = wrong (hallucination)
    assert is_correct(_ans(answer="123"), item) is False


def test_numeric_answer_correct_within_scale():
    item = GoldenItem(question="q", answer_type="number", answers=[88100000], category="retrieval")
    assert is_correct(_ans(answer="88.1"), item) is True
    assert is_correct(_ans(answer="999"), item) is False


def test_multi_valid_answer_either_matches():
    # golden accepts N/A OR 15.63
    item = GoldenItem(question="q", answer_type="number", answers=["N/A", 15.63], category="tricky")
    assert is_correct(_ans(answer="15.63"), item) is True
    assert is_correct(_ans(refused=True, answer="N/A"), item) is True


def test_name_answer_matches_golden_name():
    item = GoldenItem(question="q", answer_type="name", answers=["MITSUI O.S.K. LINES"], category="compare")
    assert is_correct(_ans(answer="The answer is MITSUI O.S.K. Lines."), item) is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_matching.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `backend/eval/matching.py`**

```python
"""Decide whether a prediction matches the golden answer.

Financial answers vary in scale (a report says "88.1" meaning 88.1 million;
the golden value is 88100000), so numeric matching tries a range of scales
within a relative tolerance. A refusal is correct exactly when "N/A" is an
acceptable golden answer.
"""

import re

from eval.golden import GoldenItem
from rag.generation.schema import RAGAnswer

_SCALES = (1, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)
_SUFFIXES = {"inc", "incorporated", "ltd", "limited", "plc", "corp", "corporation",
             "sa", "ag", "llc", "co", "company", "holdings", "group", "the"}


def parse_number(text: str) -> float | None:
    cleaned = text.replace(",", "")
    negative = "(" in cleaned and ")" in cleaned
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    value = float(match.group())
    if negative and value > 0:
        value = -value
    return value


def number_matches(pred: float, golden: float, rel_tol: float = 0.02) -> bool:
    denom = abs(golden) if golden != 0 else 1.0
    return any(abs(pred * scale - golden) <= rel_tol * denom for scale in _SCALES)


def _tokens(name: str) -> frozenset[str]:
    toks = re.findall(r"[a-z0-9]+", name.lower())
    return frozenset(t for t in toks if t not in _SUFFIXES)


def name_matches(pred: str, golden: str) -> bool:
    a, b = _tokens(pred), _tokens(golden)
    return bool(a) and bool(b) and (b <= a or a <= b)


def _is_na(value) -> bool:
    return isinstance(value, str) and value.strip().upper() == "N/A"


def is_correct(prediction: RAGAnswer, item: GoldenItem) -> bool:
    na_acceptable = any(_is_na(a) for a in item.answers)
    refused = prediction.refused or _is_na(prediction.answer)

    if refused:
        return na_acceptable

    if item.answer_type == "number":
        pred_num = parse_number(prediction.answer)
        if pred_num is None:
            return False
        return any(not _is_na(g) and number_matches(pred_num, float(g))
                   for g in item.answers if isinstance(g, (int, float)))

    # name schema
    return any(not _is_na(g) and name_matches(prediction.answer, str(g))
               for g in item.answers)
```

- [ ] **Step 4: Run to verify pass, then full suite**

Run: `uv run pytest tests/test_matching.py -v` → all pass. Then `uv run pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add backend/eval/matching.py backend/tests/test_matching.py
git commit -m "feat(eval): scale-tolerant answer matching (numeric/name/N-A)"
```

---

### Task 3: Runner (run a mode over the benchmark)

**Files:**
- Create: `backend/eval/runner.py`
- Test: `backend/tests/test_runner.py`

**Interfaces:**
- Produces:
  - `EvalRow(BaseModel)`: `question`, `category`, `answer_type`, `expected: list`, `predicted: str`, `refused: bool`, `correct: bool`, `error: str = ""`.
  - `run_mode(items, mode, deps, answer_fn=answer) -> list[EvalRow]` — call `answer_fn(item.question, mode, deps)` for each item, score with `is_correct`, capturing any per-question exception into `error` (so one failure doesn't abort the run).

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_runner.py`:
```python
"""Runner scores predictions with an injected fake answer_fn (no LLM)."""

from eval.golden import GoldenItem
from eval.runner import run_mode
from rag.generation.schema import RAGAnswer


def _items():
    return [
        GoldenItem(question="petra net income 2022?", answer_type="number",
                   answers=[88100000], category="retrieval"),
        GoldenItem(question="crossfirst liabilities 2023?", answer_type="number",
                   answers=["N/A"], category="hallucination"),
    ]


def test_run_mode_scores_each_item():
    def fake_answer(question, mode, deps):
        if "petra" in question:
            return RAGAnswer(answer="88.1", refused=False)
        return RAGAnswer(answer="N/A", refused=True)

    rows = run_mode(_items(), "hybrid", deps=None, answer_fn=fake_answer)
    assert len(rows) == 2
    assert rows[0].correct is True   # 88.1 ≈ 88,100,000
    assert rows[1].correct is True   # refusal matches N/A


def test_run_mode_captures_errors():
    def boom(question, mode, deps):
        raise RuntimeError("provider down")

    rows = run_mode(_items()[:1], "hybrid", deps=None, answer_fn=boom)
    assert rows[0].correct is False
    assert "provider down" in rows[0].error
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_runner.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `backend/eval/runner.py`**

```python
"""Run one retrieval mode over the benchmark and score each answer."""

from pydantic import BaseModel, Field

from eval.golden import GoldenItem
from eval.matching import is_correct
from rag.query import answer


class EvalRow(BaseModel):
    question: str
    category: str
    answer_type: str
    expected: list
    predicted: str
    refused: bool
    correct: bool
    error: str = ""


def run_mode(items: list[GoldenItem], mode: str, deps, answer_fn=answer) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for item in items:
        try:
            prediction = answer_fn(item.question, mode, deps)
            rows.append(EvalRow(
                question=item.question, category=item.category, answer_type=item.answer_type,
                expected=item.answers, predicted=prediction.answer,
                refused=prediction.refused, correct=is_correct(prediction, item),
            ))
        except Exception as exc:  # noqa: BLE001 — one bad question shouldn't abort the run
            rows.append(EvalRow(
                question=item.question, category=item.category, answer_type=item.answer_type,
                expected=item.answers, predicted="", refused=False,
                correct=False, error=f"{type(exc).__name__}: {exc}",
            ))
    return rows
```

- [ ] **Step 4: Run to verify pass, then full suite**

Run: `uv run pytest tests/test_runner.py -v` → 2 pass. Then `uv run pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add backend/eval/runner.py backend/tests/test_runner.py
git commit -m "feat(eval): mode runner scoring answers, resilient to errors"
```

---

### Task 4: Metrics aggregation

**Files:**
- Create: `backend/eval/metrics.py`
- Test: `backend/tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `aggregate(rows: list[EvalRow]) -> dict` — for one mode: `{"overall": {"correct", "total", "accuracy"}, "by_category": {cat: {"correct", "total", "accuracy"}}}`. `accuracy` is a float 0–1.
  - `refusal_accuracy(rows) -> float` — accuracy over the `hallucination` category only (the headline correct-refusal rate).

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_metrics.py`:
```python
"""Aggregate eval rows into per-category accuracy + headline refusal rate."""

from eval.metrics import aggregate, refusal_accuracy
from eval.runner import EvalRow


def _rows():
    return [
        EvalRow(question="a", category="retrieval", answer_type="number", expected=[1],
                predicted="1", refused=False, correct=True),
        EvalRow(question="b", category="hallucination", answer_type="number", expected=["N/A"],
                predicted="N/A", refused=True, correct=True),
        EvalRow(question="c", category="hallucination", answer_type="number", expected=["N/A"],
                predicted="5", refused=False, correct=False),
    ]


def test_aggregate_overall_and_by_category():
    agg = aggregate(_rows())
    assert agg["overall"] == {"correct": 2, "total": 3, "accuracy": 2 / 3}
    assert agg["by_category"]["hallucination"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert agg["by_category"]["retrieval"]["accuracy"] == 1.0


def test_refusal_accuracy_is_hallucination_only():
    assert refusal_accuracy(_rows()) == 0.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `backend/eval/metrics.py`**

```python
"""Aggregate scored rows into per-category and overall accuracy."""

from collections import defaultdict

from eval.runner import EvalRow


def _stats(rows: list[EvalRow]) -> dict:
    total = len(rows)
    correct = sum(1 for r in rows if r.correct)
    return {"correct": correct, "total": total,
            "accuracy": correct / total if total else 0.0}


def aggregate(rows: list[EvalRow]) -> dict:
    by_category: dict[str, list[EvalRow]] = defaultdict(list)
    for row in rows:
        by_category[row.category].append(row)
    return {
        "overall": _stats(rows),
        "by_category": {cat: _stats(rs) for cat, rs in by_category.items()},
    }


def refusal_accuracy(rows: list[EvalRow]) -> float:
    hallucination = [r for r in rows if r.category == "hallucination"]
    return _stats(hallucination)["accuracy"]
```

- [ ] **Step 4: Run to verify pass, then full suite**

Run: `uv run pytest tests/test_metrics.py -v` → 2 pass. Then `uv run pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add backend/eval/metrics.py backend/tests/test_metrics.py
git commit -m "feat(eval): per-category + overall accuracy aggregation"
```

---

### Task 5: Evaluate CLI (table + eval_results.json)

**Files:**
- Create: `backend/eval/evaluate.py`
- Modify: `backend/Makefile` (wire the `eval` target)
- Test: `backend/tests/test_evaluate.py`

**Interfaces:**
- Produces:
  - `build_report(items, modes, deps, answer_fn=answer) -> dict` — run each mode, aggregate, and assemble `{"modes": {mode: aggregate}, "refusal_accuracy": {mode: float}, "rows": {mode: [EvalRow...]}}`.
  - `render_table(report) -> str` — a compact per-category × per-mode accuracy table (plain text).
  - `write_results(report, path) -> None` — dump the report to `eval_results.json` (UTF-8).
  - `main()` — CLI: `build_deps()`, run modes `basic, hybrid, agentic`, print the table, write `data/eval_results.json`. UTF-8 stdout.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_evaluate.py`:
```python
"""Report assembly + rendering + JSON writing (fake answer_fn, no LLM)."""

import json

from eval.evaluate import build_report, render_table, write_results
from eval.golden import GoldenItem
from rag.generation.schema import RAGAnswer


def _items():
    return [
        GoldenItem(question="petra 2022?", answer_type="number", answers=[88100000],
                   category="retrieval"),
        GoldenItem(question="crossfirst 2023?", answer_type="number", answers=["N/A"],
                   category="hallucination"),
    ]


def _answer_fn(question, mode, deps):
    if "petra" in question:
        return RAGAnswer(answer="88.1", refused=False)
    return RAGAnswer(answer="N/A", refused=True)


def test_build_report_covers_each_mode():
    report = build_report(_items(), ["basic", "hybrid"], deps=None, answer_fn=_answer_fn)
    assert set(report["modes"]) == {"basic", "hybrid"}
    assert report["modes"]["hybrid"]["overall"]["accuracy"] == 1.0
    assert report["refusal_accuracy"]["hybrid"] == 1.0


def test_render_table_mentions_modes_and_categories():
    report = build_report(_items(), ["hybrid"], deps=None, answer_fn=_answer_fn)
    table = render_table(report)
    assert "hybrid" in table
    assert "hallucination" in table


def test_write_results_roundtrips(tmp_path):
    report = build_report(_items(), ["hybrid"], deps=None, answer_fn=_answer_fn)
    out = tmp_path / "eval_results.json"
    write_results(report, str(out))
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["modes"]["hybrid"]["overall"]["total"] == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_evaluate.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `backend/eval/evaluate.py`**

```python
"""Run the benchmark across modes; print a table and write eval_results.json."""

import json
import sys
from pathlib import Path

from core.config import get_settings
from eval.golden import load_golden
from eval.metrics import aggregate, refusal_accuracy
from eval.runner import run_mode
from rag.query import answer, build_deps

_MODES = ["basic", "hybrid", "agentic"]


def build_report(items, modes, deps, answer_fn=answer) -> dict:
    report = {"modes": {}, "refusal_accuracy": {}, "rows": {}}
    for mode in modes:
        rows = run_mode(items, mode, deps, answer_fn=answer_fn)
        report["modes"][mode] = aggregate(rows)
        report["refusal_accuracy"][mode] = refusal_accuracy(rows)
        report["rows"][mode] = [r.model_dump() for r in rows]
    return report


def render_table(report: dict) -> str:
    modes = list(report["modes"])
    categories = sorted({c for m in modes for c in report["modes"][m]["by_category"]})
    width = 16
    header = "category".ljust(width) + "".join(m.ljust(width) for m in modes)
    lines = [header, "-" * len(header)]
    for cat in categories:
        row = cat.ljust(width)
        for mode in modes:
            stats = report["modes"][mode]["by_category"].get(cat)
            cell = f"{stats['accuracy']:.0%} ({stats['correct']}/{stats['total']})" if stats else "-"
            row += cell.ljust(width)
        lines.append(row)
    overall = "OVERALL".ljust(width) + "".join(
        f"{report['modes'][m]['overall']['accuracy']:.0%}".ljust(width) for m in modes)
    lines += ["-" * len(header), overall]
    return "\n".join(lines)


def write_results(report: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    settings = get_settings()
    items = load_golden(str(Path(settings.data_dir).parent / "benchmark" / "answers.json"))
    deps = build_deps(settings)
    report = build_report(items, _MODES, deps)
    print(render_table(report))
    out = str(Path(settings.data_dir).parent / "eval_results.json")
    write_results(report, out)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Wire the Makefile `eval` target**

In `backend/Makefile`, replace the placeholder `eval` recipe with:
```make
eval:
	PYTHONUTF8=1 uv run python -m eval.evaluate
```

- [ ] **Step 5: Run to verify pass, then full suite**

Run: `uv run pytest tests/test_evaluate.py -v` → 3 pass. Then `uv run pytest -q` → full suite green.

- [ ] **Step 6: Commit**

```bash
git add backend/eval/evaluate.py backend/Makefile backend/tests/test_evaluate.py
git commit -m "feat(eval): evaluate CLI — per-category/mode table + eval_results.json"
```

- [ ] **Step 7: Real eval run (manual — needs GROQ_API_KEY + built index)**

Not a test. Run once by the controller to produce the real numbers (≈40 questions × 3 modes × 2 Groq calls = ~240 calls; slow on the free tier — the runner tolerates rate-limit errors per question, and a re-run is cheap since answers are deterministic-ish). Review the printed table and `data/eval_results.json`:
```bash
cd backend && PYTHONUTF8=1 uv run python -m eval.evaluate
```
Note `eval_results.json` goes under `data/` — decide whether to commit it (it's the dashboard's data source; committing the latest run is reasonable, and `.gitignore` currently does NOT ignore it).

---

## Self-Review

**Spec coverage (Plan 5 slice of the design §3.7):**
- Golden loader from `answers.json` (question/schema/answers/category) → Task 1. ✅
- Matching: numeric tolerance **+ unit/scale normalization** (88.1M ≈ 88,100,000, surfaced by the smoke test), `N/A` handling, multi-valid-answer support, name matching → Task 2. ✅
- Runner over `answer(question, mode)`, resilient to per-question errors → Task 3. ✅
- Per-category × per-mode aggregation + headline correct-refusal rate → Task 4. ✅
- CLI: comparison table + committed `eval_results.json` (the dashboard's data source) → Task 5. ✅
- LangSmith dataset/experiment path: **deferred** (documented enhancement; the local harness is the primary, offline-testable artifact and what the dashboard reads).

**Placeholder scan:** none — full code in every step.

**Type consistency:** `GoldenItem` (Task 1) is consumed by `matching.is_correct` (Task 2), `runner.run_mode` (Task 3). `RAGAnswer` (from `rag.generation.schema`) is the prediction type in matching + runner. `EvalRow` (Task 3) is consumed by `metrics.aggregate` (Task 4) and `evaluate.build_report` (Task 5). `run_mode(items, mode, deps, answer_fn=answer)` signature (Task 3) matches its call in `build_report` (Task 5). The real `answer`/`build_deps` come from `rag.query` (Plan 3).
