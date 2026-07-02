# Financial-Report RAG

Hallucination-resistant Retrieval-Augmented Generation over SEC 10-K filings,
evaluated on an adversarial benchmark. See `../docs/superpowers/specs/` for the
design and `../docs/superpowers/plans/` for the build plans.

## Quickstart

```bash
cd backend
uv sync
cp .env.example .env      # add your GROQ_API_KEY
pytest                    # run the test suite
```

More commands (ingest, run, eval) arrive in later build plans.
