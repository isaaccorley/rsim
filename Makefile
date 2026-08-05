.PHONY: install check clean

install:
	uv sync --all-groups
	uv run pre-commit install

check:
	uv run pre-commit run --all-files
	uv run pytest -vvv --cov=src
	uv run ty check

clean:
	rm -rf .pytest_cache .ruff_cache .typos .mypy_cache .hypothesis .venv/__pycache__
	find src scripts tests -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf outputs
	rm -f texput.log
