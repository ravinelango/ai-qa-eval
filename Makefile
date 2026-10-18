# Every major action has a target. Targets for later phases fail loudly with the
# phase that will implement them, so nothing looks finished before it is.

UV ?= uv
.DEFAULT_GOAL := help

define not_yet
	@echo "make $@: not yet implemented ($(1))" >&2; exit 2
endef

.PHONY: help
help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

# --- Phase 0: works today, offline, no keys -------------------------------------

.PHONY: setup
setup: ## Install from uv.lock and install git hooks
	$(UV) sync --locked
	$(UV) run pre-commit install

.PHONY: lint
lint: ## Ruff lint + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

.PHONY: fmt
fmt: ## Auto-format and apply safe lint fixes
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

.PHONY: typecheck
typecheck: ## mypy (strict)
	$(UV) run mypy

.PHONY: test
test: ## Unit + offline integration tests (sockets disabled; live tests skipped)
	$(UV) run pytest

.PHONY: check
check: lint typecheck test ## Everything ordinary CI runs

.PHONY: demo
demo: ## Offline fake-provider demo (synthetic output, no keys, no network)
	$(UV) run python -m apps.core.demo

.PHONY: build
build: ## Build the wheel and verify its packages
	$(UV) build --wheel --out-dir dist
	$(UV) run python scripts/check_wheel.py dist/*.whl

.PHONY: test-live
test-live: ## PAID: one real API call (needs ANTHROPIC_API_KEY)
	$(UV) run pytest tests/live --run-live

.PHONY: clean
clean: ## Remove caches, build output and local demo runs
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist runs/demo

# --- Phase 1a: RAG corpus, index and assistant (no API calls) --------------------

.PHONY: corpus-refresh model-fetch ingest ingest-check retrieval-smoke test-model
corpus-refresh: ## NETWORK: re-download corpus sources into data/corpus (review the diff)
	$(UV) run python -m apps.rag.corpus_fetch
model-fetch: ## NETWORK: download the pinned embedder and reranker into .cache/ (never committed)
	$(UV) run python -m apps.rag.model_fetch
ingest: ## Frozen corpus -> chunk manifest -> embeddings -> local Chroma (no network)
	$(UV) run python -m apps.rag.ingest
ingest-check: ## Check the saved chunk manifest is reproduced byte for byte
	$(UV) run python -m apps.rag.ingest --check
retrieval-smoke: ## Developer-authored retrieval check on the real index (not an eval); SMOKE_ARGS="--mode ... --spec ..."
	$(UV) run python -m apps.rag.smoke $(SMOKE_ARGS)
test-model: ## Tests that load the pinned local embedding model (no network)
	$(UV) run pytest tests/model --run-model

# --- Later phases --------------------------------------------------------------

.PHONY: label eval eval-smoke judge-validate redteam experiment dashboard
label: ## Labeling CLI for the golden dataset
	$(call not_yet,Phase 1b)
eval: ## Full golden-dataset eval run
	$(call not_yet,Phase 1c)
eval-smoke: ## Smoke subset used by the CI gate
	$(call not_yet,Phase 1e)
judge-validate: ## Judge-vs-human agreement report
	$(call not_yet,Phase 1d)
redteam: ## Hand-authored cases + promptfoo + garak
	$(call not_yet,Phase 2)
experiment: ## Run the dataset against a named variant
	$(call not_yet,Phase 3a)
dashboard: ## Streamlit model-comparison dashboard
	$(call not_yet,Phase 3c)
