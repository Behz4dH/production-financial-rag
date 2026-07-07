.PHONY: ingest run ui eval test

ingest:
	PYTHONUTF8=1 uv run python -m rag.ingestion.pipeline

test:
	uv run pytest -q

run:
	uv run uvicorn "app.main:create_app" --factory --reload --port 8000

eval:
	@echo "Eval harness arrives in a later plan"
