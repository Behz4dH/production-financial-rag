.PHONY: ingest run ui eval test

ingest:
	PYTHONUTF8=1 uv run python -m rag.ingestion.pipeline

test:
	uv run pytest -q

run:
	@echo "API server arrives in a later plan"

eval:
	@echo "Eval harness arrives in a later plan"
