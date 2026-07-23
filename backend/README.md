# Financial-Report RAG

Hallucination-resistant Retrieval-Augmented Generation over SEC 10-K filings,
evaluated on an adversarial benchmark. See `../docs/superpowers/specs/` for the
design and `../docs/superpowers/plans/` for the build plans.

## Quickstart

```bash
cd backend
uv sync
cp .env.example .env      # add your GROQ_API_KEY
make ingest               # build the index from data/docs (metadata is cached)
make run                  # API on http://localhost:8000  (Swagger at /docs)
make eval                 # run the benchmark -> per-category table + eval_results.json
make test                 # pytest
```

Endpoints: `POST /chat {message, thread_id?, mode?}`, `GET /health`, `GET /metrics`.

## Eval regression gate

The benchmark doubles as a release gate: `eval/baseline.json` (committed) holds
the accepted per-mode quality bar, and the gate fails when a fresh run regresses.

```bash
make eval                 # run the benchmark -> data/eval_results.json
make gate                 # compare against eval/baseline.json (exit 1 on regression)
make baseline             # accept the current results as the new bar (commit the diff)
```

Gate rules: refusal accuracy (hallucination resistance) may not drop at all;
overall accuracy may drop at most 3% per mode (one flipped question of 40 —
slack for LLM nondeterminism). Improvements pass; committing the updated
baseline is how the bar gets raised, so a raise is itself code-reviewed.

This is currently a local gate (`make gate`), run by hand before merging
pipeline-affecting changes. Wiring it into CI (a workflow that reruns
`make eval && make gate` on PRs touching `rag/` or `eval/`, gated on a
`GROQ_API_KEY` repository secret) is a natural next step, not yet done.

## Dashboard quickstart

With the API running (`make run`, terminal 1), start the frontend in a second terminal:

```bash
cd backend
make ui                   # installs deps and starts the Vite dev server on http://localhost:5173
```

Open `http://localhost:5173`. The **Chat** tab lets you ask a question and expand the pipeline
trace panel to see each step the request went through — entity resolution -> retrieval ->
reranking -> generation (or a short trace ending in a refusal, with the reason, for
out-of-corpus or wrong-year questions). The **Eval Results** tab shows the benchmark table from
the most recent `make eval` run; it requires `make eval` to have been run at least once so that
`data/eval_results.json` exists.
