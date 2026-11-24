.PHONY: install data train serve smoke test lint format compile docker docker-run clean

PORT ?= 8080
MODEL_DIR ?= artifacts/model

install:
	pip install -e ".[dev,serve,gcp]"

data:
	./scripts/download_data.sh

train: data
	churn-train --data data/telco_churn.csv --out $(MODEL_DIR)

serve:
	MODEL_DIR=$(MODEL_DIR) uvicorn churn.serve.app:app --port $(PORT) --reload

smoke:
	./scripts/smoke_test.sh http://localhost:$(PORT)

test:
	pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

compile:
	python -m pipelines.churn_pipeline --compile-only

docker:
	docker build -f docker/serving.Dockerfile -t churn-serving:local .

docker-run: docker
	docker run --rm -p $(PORT):8080 -v $(PWD)/$(MODEL_DIR):/model -e MODEL_DIR=/model churn-serving:local

clean:
	rm -rf artifacts churn_pipeline.yaml .pytest_cache .ruff_cache .coverage
	find . -name __pycache__ -type d -exec rm -rf {} +
