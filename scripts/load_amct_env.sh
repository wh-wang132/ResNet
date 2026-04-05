#!/bin/sh

REPO_ROOT=${REPO_ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}

. "$REPO_ROOT/scripts/load_cann_env.sh"

export PYTHONPATH="$REPO_ROOT/src"
