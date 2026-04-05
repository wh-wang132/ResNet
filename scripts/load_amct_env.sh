#!/bin/sh

REPO_ROOT=${REPO_ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}
PIXI_ENV_PREFIX="$REPO_ROOT/.pixi/envs/default"
PIXI_LIB="$PIXI_ENV_PREFIX/lib"

. "$REPO_ROOT/scripts/load_cann_env.sh"

if [ -d "$PIXI_LIB" ]; then
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
        export LD_LIBRARY_PATH="$PIXI_LIB:$LD_LIBRARY_PATH"
    else
        export LD_LIBRARY_PATH="$PIXI_LIB"
    fi
fi

export PYTHONPATH="$REPO_ROOT/src"
