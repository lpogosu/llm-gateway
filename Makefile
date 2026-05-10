SHELL := /bin/bash
PYTHON ?= python
COMPOSE ?= docker compose
GATEWAY_URL ?= http://localhost:8080
API_KEY ?= sk-local-dev
HELM_IMAGE ?= alpine/helm:latest

.DEFAULT_GOAL := help
.PHONY: help install up down logs test lint fmt typecheck bench demo helm-lint helm-template docker-build ci clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with its development extras
	$(PYTHON) -m pip install -e ".[dev]"

up: ## Start gateway, Redis, Prometheus and Grafana
	$(COMPOSE) up -d --build
	@echo "gateway    $(GATEWAY_URL)"
	@echo "prometheus http://localhost:9090"
	@echo "grafana    http://localhost:3000 (anonymous viewer)"

down: ## Stop the stack and drop its volumes
	$(COMPOSE) down -v

logs: ## Follow the gateway log
	$(COMPOSE) logs -f gateway

test: ## Run the test suite with coverage
	$(PYTHON) -m pytest --cov=app --cov-report=term-missing --cov-report=xml

lint: ## Run ruff
	ruff check .

fmt: ## Apply the fixes ruff can apply
	ruff check --fix .

typecheck: ## Run mypy in strict mode
	mypy --strict app

bench: ## Measure gateway overhead against a stub upstream (no GPU needed)
	$(PYTHON) scripts/bench.py --requests 500 --concurrency 20

demo: ## Walk through cache, routing, rate limiting and usage against a running stack
	GATEWAY_URL=$(GATEWAY_URL) API_KEY=$(API_KEY) bash scripts/demo.sh

helm-lint: ## Lint the chart in a container, no local helm needed
	docker run --rm -v "$(CURDIR)":/w -w /w $(HELM_IMAGE) lint charts/llm-gateway

helm-template: ## Render the chart with every optional object enabled
	docker run --rm -v "$(CURDIR)":/w -w /w $(HELM_IMAGE) template gw charts/llm-gateway \
		--set secrets.create=true --set ingress.enabled=true --set serviceMonitor.enabled=true

docker-build: ## Build the runtime image
	docker build -t llm-gateway:local .

ci: lint typecheck test helm-lint ## Everything the pipeline runs

clean: ## Remove build and tooling artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml .coverage build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
