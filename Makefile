.PHONY: setup data features train-baseline train-gbm train-emb eval serve test lint fmt docker clean

PY ?= python
UV ?= uv

setup:
	$(UV) venv
	$(UV) pip install -e ".[dev]"

data:
	$(PY) -m rossmann_forecast.cli download

features:
	$(PY) -m rossmann_forecast.cli build-features

train-baseline:
	$(PY) -m rossmann_forecast.cli train-baseline

train-gbm:
	$(PY) -m rossmann_forecast.cli train-gbm

train-emb:
	$(PY) -m rossmann_forecast.cli train-emb

eval:
	$(PY) -m rossmann_forecast.cli evaluate

serve:
	uvicorn rossmann_forecast.serve.api:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q

lint:
	ruff check src tests

fmt:
	ruff format src tests
	ruff check --fix src tests

docker:
	docker build -t rossmann-forecast:latest .

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov mlruns artifacts
