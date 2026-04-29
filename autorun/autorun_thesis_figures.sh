#!/bin/sh
set -eu

. "$REPO_ROOT/scripts/_require_public_env.sh"
require_public_env

cd "$REPO_ROOT"

uv run python -m thesis_figures \
    --output_root output \
    --formats png,svg \
    --strict
