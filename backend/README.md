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
