#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest -q
