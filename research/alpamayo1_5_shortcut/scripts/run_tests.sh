#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
readonly RECIPE_DIR="$REPO_ROOT/recipes/alpamayo1_5_sft"
readonly ENV_DIR="${ALPAMAYO_ENV:-$RECIPE_DIR/a1_5_sft}"
readonly PYTHON_BIN="${PYTHON_BIN:-$ENV_DIR/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python environment not found: %s\n' "$PYTHON_BIN" >&2
  printf 'Set ALPAMAYO_ENV or follow recipes/alpamayo1_5_sft/README.md.\n' >&2
  exit 2
fi

cd "$RECIPE_DIR"
"$PYTHON_BIN" -m pytest -q \
  tests/test_shortcut_modules.py \
  tests/test_shortcut_model_config.py \
  tests/test_pai_trajectory.py \
  tests/test_paper_shortcut.py \
  tests/test_paper_empirical_ablation.py

"$PYTHON_BIN" "$SCRIPT_DIR/test_fill_missing_evaluations.py"
