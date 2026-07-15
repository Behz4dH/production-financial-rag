.PHONY: ingest run ui eval test gate baseline

ingest:
	PYTHONUTF8=1 uv run python -m rag.ingestion.pipeline

test:
	uv run pytest -q

run:
	uv run uvicorn "app.main:create_app" --factory --reload --port 8000

eval:
	PYTHONUTF8=1 uv run python -m eval.evaluate

# Tolerance = one flipped question of the CURRENT benchmark size (10-item
# slice -> 0.10). Restore the default 0.03 with the full 40-question set.
gate:
	PYTHONUTF8=1 uv run python -m eval.gate --overall-tolerance 0.10

baseline:
	PYTHONUTF8=1 uv run python -m eval.gate --update-baseline

ui:
	cd ../frontend && npm install && npm run dev
