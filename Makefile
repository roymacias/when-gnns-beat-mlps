# ============================================================================
# Variables & Environment
# ============================================================================

# Force Python into UTF-8 mode to prevent Windows CP1252 decoding issues in third-party libraries.
export PYTHONUTF8 = 1

# Display available commands by default.
.DEFAULT_GOAL := help

# Execute commands inside the managed virtual environment.
UV := uv run

# ============================================================================
# GNNBench Commands
# ============================================================================

.PHONY: help setup format lint typecheck test clean download splits data node link graph all

help:
	@echo "GNNBench Commands"
	@echo "-----------------"
	@echo ""
	@echo "Development"
	@echo "  setup        Install project dependencies."
	@echo "  format       Format the source code."
	@echo "  lint         Run linting and verify formatting."
	@echo "  typecheck    Run static type checking."
	@echo "  test         Run the test suite."
	@echo "  clean        Remove Python caches and temporary files."
	@echo ""
	@echo "ML Pipelines"
	@echo ""
	@echo "  download     Download benchmark datasets."
	@echo "  splits       Generate reproducible dataset splits."
	@echo "  data         Run download and split generation."
	@echo ""
	@echo "  node         Train and evaluate node classification."
	@echo "  link         Train and evaluate link prediction."
	@echo "  graph        Train and evaluate graph classification."
	@echo ""
	@echo "  all          End-to-end benchmark execution."

# ----------------------------------------------------------------------------
# Development
# ----------------------------------------------------------------------------

setup:
	uv sync --frozen

format:
	$(UV) ruff format src/ tests/

lint:
	$(UV) ruff check src/ tests/
	$(UV) ruff format --check src/ tests/

typecheck:
	$(UV) mypy

test:
	$(UV) pytest

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .ruff_cache/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# ----------------------------------------------------------------------------
# ML Pipelines
# ----------------------------------------------------------------------------

# Data
download:
	$(UV) python -m gnnbench.data.download

splits:
	$(UV) python -m gnnbench.data.splits

data: download splits

# Training and Evaluation
node:
	$(UV) python -m gnnbench.training.node --config config/node.yaml
	$(UV) python -m gnnbench.evaluation.node --config config/node.yaml

link:
	$(UV) python -m gnnbench.training.link --config config/link.yaml
	$(UV) python -m gnnbench.evaluation.link --config config/link.yaml

graph:
	$(UV) python -m gnnbench.training.graph --config config/graph.yaml
	$(UV) python -m gnnbench.evaluation.graph --config config/graph.yaml

# End-to-End Pipeline
all: data node link graph
