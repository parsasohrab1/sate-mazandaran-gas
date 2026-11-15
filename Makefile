# ============================================================
#  Sate Mazandaran Gas — AI Weld Defect Detection
#  Makefile (CMMI ML5 build automation)
#  Petro Pala Toos / Dr. N. Ashrafi-Payaman
# ============================================================

SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help
PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip
PKG    := weld_defect

.PHONY: help setup dev lint type test test-fast test-cov security format \
        data data-synthetic train evaluate predict serve mlflow \
        docker-build docker-up docker-down \
        docs docs-serve clean release

help: ## Show this help
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	/^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Create virtualenv and install runtime deps
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && $(PIP) install --upgrade pip && $(PIP) install -r requirements.txt

dev: setup ## Install dev/test deps on top of runtime
	. .venv/bin/activate && $(PIP) install -r requirements-dev.txt && pre-commit install

lint: ## Run ruff
	. .venv/bin/activate && ruff check src tests

format: ## Run black + ruff --fix
	. .venv/bin/activate && black src tests && ruff check --fix src tests

type: ## Run mypy
	. .venv/bin/activate && mypy src

test: ## Run full test suite with coverage
	. .venv/bin/activate && pytest

test-fast: ## Run tests excluding slow/gpu/integration
	. .venv/bin/activate && pytest -m "not slow and not gpu and not integration"

test-cov: ## Run tests and open HTML coverage
	. .venv/bin/activate && pytest && xdg-open htmlcov/index.html || true

security: ## Run bandit
	. .venv/bin/activate && bandit -r src -c pyproject.toml

data: ## Download real data from catalogued sources (see data/raw/README.md)
	. .venv/bin/activate && python scripts/download_data.py

data-synthetic: ## Generate synthetic NDT images for first validation
	. .venv/bin/activate && python scripts/generate_synthetic_data.py --n-samples 1200 --out data/synthetic

train: ## Train the selected model (defaults to ensemble)
	. .venv/bin/activate && python -m weld_defect.cli train --config configs/train.yaml

evaluate: ## Evaluate a trained model on the held-out test set
	. .venv/bin/activate && python -m weld_defect.cli evaluate --config configs/eval.yaml

predict: ## Predict on a single image
	. .venv/bin/activate && python -m weld_defect.cli predict --image $$IMG --model models/ensemble_best.pt

serve: ## Serve the REST API (uvicorn)
	. .venv/bin/activate && uvicorn weld_defect.api.app:app --host 0.0.0.0 --port 8000

mlflow: ## Start the MLflow tracking UI
	. .venv/bin/activate && mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5000

docker-build: ## Build all images
	docker compose -f devops/docker-compose.yml build

docker-up: ## Bring the full stack up
	docker compose -f devops/docker-compose.yml up -d

docker-down: ## Tear the stack down
	docker compose -f devops/docker-compose.yml down

docs: ## Build MkDocs site
	. .venv/bin/activate && mkdocs build

docs-serve: ## Serve docs locally
	. .venv/bin/activate && mkdocs serve

clean: ## Remove build/test artifacts
	rm -rf build dist *.egg-info .pytest_cache .tox .coverage htmlcov coverage.xml \
	       .mypy_cache .ruff_cache mlruns wandb
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

release: ## Cut a release tag (usage: make release V=1.0.0)
	@test -n "$(V)" || (echo "usage: make release V=1.0.0" && exit 1)
	git tag -a v$(V) -m "Release v$(V)"
	git push origin v$(V)
