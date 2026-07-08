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
