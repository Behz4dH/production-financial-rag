# Engineering Notes — incidents, dead ends, and why the code is shaped this way

A decision log preserved out of the code comments. Each entry records what
happened, what was measured, and what the code does about it. Dates refer to
the July 2026 build; commits on `build/multi-source-decomposition` carry the
detailed diffs.

## Parsing & chunking

**The geometric table tower (retired 2026-07-15, commit bb3f1ad).**
The original loader used PyMuPDF `find_tables()` (line-based). A corpus census
showed ~7 of 20 filings were effectively table-blind — borderless layouts
(TransUnion 10-K: 11 tables detected in 167 pages; First Mid: 2 in 110).
A `strategy="text"` fallback recovered them but over-boxed prose, so page
headers ("146 Petra Diamonds — Annual Report 2022") became "tables" whose junk
rows flooded BM25 with exactly the tokens every query contains (company name +
year). A financial-number guard fixed that, but extraction stayed ragged —
`Total intangible ass -- ts: ets . . . $5,67 9. 5` — and generation correctly
refused figures it couldn't read. The stack of heuristics (detection strategy,
genuineness filter, numeric-density guard, financial-number regex, bbox
subtraction) was replaced wholesale by one pymupdf4llm markdown pass. The same
TransUnion row now renders as
`|Total intangible assets|$5,944.1|$(2,268.6)|$3,675.5|$5,679.5|$(1,908.9)|$3,770.6|`.

**Why table rows are atomic, caption'd, term-dense units.**
Whole-table chunks were prototyped and scored 0/5 on retrieval: a 1,100-char
balance sheet mentions the query's words once or twice among hundreds of
number tokens, so BM25's frequency/length normalization buries it under prose.
Row-level units are the lexical fix; the caption + column headers carry the
units and period context whose absence caused a fabrication bug (parent-only
vs consolidated confusion) and a class of correct-but-needless refusals.

**Why running headers/footers are stripped.**
Boilerplate-led chunks pushed figures past the reranker's 500-char snippet
window and fed BM25 junk tokens. Measured directly: the 8b reranker's
all-zeros collapse (below) was triggered by batches of boilerplate-led
snippets.

**Caption-prefixing history.** Prefixing rows with captions was originally
tried against the cross-encoder and made ranking WORSE (Tradition +0.969 →
+0.152). The conclusion "captions don't help" was reranker-specific: with the
LLM reranker and clean markdown rows, captions are a clear win. Beware
conclusions bound to a component that later gets replaced.

**Basic-mode regression on migration (accepted 2026-07-15).**
The markdown migration dropped basic (vector-only) mode 90%→70% while hybrid
held 100%. Root cause: the geometric parser's ragged extraction *duplicated*
figures into prose that vector search happened to surface — a bug acting as
an accidental feature. Proper row extraction removed the duplication, exposing
basic mode's honest ceiling and making the basic-vs-hybrid gap a true
measurement of BM25's contribution. Baseline updated deliberately.

## Retrieval & reranking

**Cross-encoders bury financial data under fluent prose (measured twice).**
`bge-reranker-base` ranked a director's employment bio above the shareholders'
equity row (~20x score gap). The 5x bigger `bge-reranker-v2-m3` failed
identically on the same pools (1/5 survived top-5 vs the LLM reranker's 4/4):
its #1 picks were a corporate slogan, "Major shareholders" prose, and the
chairman's letter. The bias is the model family — topical fluency over answer
presence — not the parameter count. Hence the LLM reranker, whose rubric asks
the one question that matters: does this excerpt contain the asked-for figure?

**The 8b all-zeros collapse (2026-07-13).**
`llama-3.1-8b-instant` scoring 20 candidates in one call emitted a well-formed
JSON array of all 0.0 — including for a chunk it scored 1.0 in isolation.
Batch-size sweep: 20 → all zeros, 10 → nearly all, 5 → sane. Classic small-
model repetition collapse when no clearly-relevant snippet anchors the batch.
Because the rerank score doubles as the refusal gate, one degenerate batch
refused the entire question — the UI refused *everything* for a day. Fix: the
reranker runs on its own stronger model (`reranker_llm_model`), never the 8b.

**Rank vs absolute score — how a validated prototype still broke production.**
The LLM-rerank probe validated *ranking* (ties broken by input order hid the
all-zeros case) while production wired the *absolute* score into the refusal
threshold. The probe's own caveat predicted the failure ("a naive swap would
have refused the Petra question"), but the swap shipped without the flagged
refusal-signal redesign. Lesson: a harness validates exactly the metric it
measures, nothing more.

**Fusion weights are a dead knob (measured 2026-07-14).**
EnsembleRetriever pool membership is identical across 0.2/0.8, 0.4/0.6,
0.8/0.2 on every test case — weights only reorder the union of the two arms'
top-k lists, and the reranker rescores the whole pool anyway. The levers that
matter are per-arm `top_k` (membership) and query vocabulary (synonym gaps).

**The vocabulary gap.** "Net income" shares zero tokens with "Profit for the
Year"; bge-small doesn't bridge it either. Whichever pipeline won a given case
often did so via a lucky lexical carrier. Known fix (not yet built): financial
metric synonym expansion at query time.

## Generation

**Retrieve small, generate big (parent-page expansion).**
Reranked chunks are precise enough to find but often lack the page's labeling
— column headers, units notes live in sibling chunks — so strict-match
generation refused with the answer literally in front of it ("88.1 is a
possible match but the year cannot be confirmed"). Expanding kept chunks to
their full pages fixed the false-refusal class. The refusal gate still judges
the precise chunks; only generation sees pages.

**The 8b refuses verbatim answers (retired from generation 2026-07-14).**
Given context stating "net sales … to $688.4 million" at rerank score 1.0, the
8b quoted the figure in its reasoning and refused anyway. Scout and
gpt-oss-120b answered the identical context correctly. The strict matching
procedure requires judgment the 8b doesn't reliably have.

**Source-blind context trimming lost a whole company (compare bug).**
On "Holley vs Safe & Green", token-budget trimming dropped every page of the
lower-ranked company; the model correctly refused a one-sided comparison. Fix:
multi-source trimming keeps at least one page per source. Same crowding-out
disease that was fixed at retrieval (per-source budgets), one layer deeper.

**Hash filenames made excerpts unattributable (compare bug #2).**
With S&G's pages guaranteed, the model then claimed "no information for
Holley" — while Holley's row sat in context labeled `194000c9….pdf`. Sources
are opaque hashes; on compares the model must map excerpt→company. Fix:
company names stamped on excerpts at query time from the entity index (never
persisted onto chunks — single-source-of-truth rule).

## Provider quirks (Groq)

**Token budget vs the 6,000 TPM ceiling.** A single request above the
per-minute token cap is a hard 413 no retry can fix (11 dead benchmark
questions). The budget covers question+context only; system prompt + tool
schema add ~1,000 tokens. Additionally cl100k (tiktoken) undercounts
Llama-family tokenization on numeric-dense text by up to ~25% — hence the
margin multiplier on all budget comparisons.

**`tool_use_failed` recoveries.** Groq validates tool arguments strictly and
returns the raw payload in the error body. Observed rejects that contained
*correct answers*: the 8b's `<function=AnswerDraft> {…}` wrapper; Scout's
`[{"name", "parameters"}]` envelope with string booleans (`"refused":
"false"`) and stringified arrays (`"companies": "[]"`). One eval run discarded
6 correct answers over these quibbles. All structured calls now share one
lenient recovery path (`invoke_structured`).

**Daily quota choreography.** Free-tier TPD limits: 8b 500K, 70b only 100K —
the 70b reranker at ~4k tokens/question capped the system at ~25 questions/day,
which mimicked mysterious mid-eval failures. Scout (500K TPD, 30K TPM) and
gpt-oss-120b (separate 200K bucket) rebalanced the budget; distributing stages
across models multiplies effective daily quota.

## Evaluation

**Errors masquerading as quality.** A 60%-scoring eval turned out to be 27-32
rate-limited questions per mode with *zero wrong answers among survivors*
(21/21 correct). Eval accuracy without an error breakdown is a lie; the runner
now self-paces on 429s and errors are reported separately.

**Alias blindness.** "MOL Group" scored wrong against golden "MITSUI O.S.K.
LINES" — same company. Name matching now bridges through each filing's alias
group from the entity metadata.

**The agentic mode (deleted before this log began).** A LangGraph retry loop
scored identically to plain hybrid (85%) because its "self-correcting" retry
never rewrote the query — it re-ran the identical search. Deleted. A codebase
is measured by what you can justify keeping.

## Security

**Output PII masking redacted financial figures.** The phone-number regex
matched any plain 10-digit run — `mask_output("Shares outstanding were
1234567890")` → `[PHONE REDACTED]`. In a system whose answers are made of
numbers, output masking destroyed the product. Masking is input-only; answers
derive from public filings.
