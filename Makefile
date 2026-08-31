.PHONY: install lint format typecheck test check pipeline

install:  ## Sync the environment from the lockfile, including dev tools
	uv sync --group dev
	uv run pre-commit install

lint:  ## Check style/import-order/likely-bug rules
	uv run ruff check .

format:  ## Auto-format code
	uv run ruff format .

typecheck:  ## Static type-check src/
	uv run mypy

test:  ## Run the test suite
	uv run python -m pytest -v

check: lint typecheck test  ## Everything CI will run

pipeline:  ## Fetch data, engineer features, write to SQLite (needs real internet access)
	uv run python -m cryptopulse.pipeline

train:  ## Train the model and write a checkpoint
	uv run python -m cryptopulse.training.train