#!/usr/bin/env bash
# Format, lint, type-check and test the Python bindings. Fails on the first error.
#
#   scripts/check-python.sh
#
# Expects the dev environment (`uv sync`, see bindings/python/README.md).
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== ruff format --check"
uv run ruff format --check bindings/python/fastatoms tests/py

echo "== ruff check"
uv run ruff check bindings/python/fastatoms tests/py

echo "== mypy --strict"
uv run mypy

echo "== pytest"
uv run pytest tests/py -q
