#!/bin/sh

REPO_ROOT=${REPO_ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}
PIXI_ENV_PREFIX="$REPO_ROOT/.pixi/envs/default"
PIXI_BIN="$PIXI_ENV_PREFIX/bin"
PIXI_LIB="$PIXI_ENV_PREFIX/lib"
CANN_ROOT="$PIXI_ENV_PREFIX/Ascend/cann-8.5.0"

. "$REPO_ROOT/scripts/load_cann_env.sh"

if [ -d "$PIXI_BIN" ]; then
    export PATH="$PIXI_BIN:$PATH"
fi

if [ -d "$PIXI_LIB" ]; then
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
        export LD_LIBRARY_PATH="$PIXI_LIB:$LD_LIBRARY_PATH"
    else
        export LD_LIBRARY_PATH="$PIXI_LIB"
    fi
fi

export PYTHONPATH="$CANN_ROOT/python/site-packages:$CANN_ROOT/opp/built-in/op_impl/ai_core/tbe"
