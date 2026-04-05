#!/bin/sh

REPO_ROOT=${REPO_ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}
PIXI_ENV_PREFIX="$REPO_ROOT/.pixi/envs/default"
PIXI_BIN="$PIXI_ENV_PREFIX/bin"
PIXI_LIB="$PIXI_ENV_PREFIX/lib"
CANN_ROOT="$PIXI_ENV_PREFIX/Ascend/cann-8.5.0"
PIXI_GXX_PREFIX=""

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

for candidate in "$PIXI_ENV_PREFIX"/lib/gcc/x86_64-conda-linux-gnu/*; do
    if [ -d "$candidate" ]; then
        PIXI_GXX_PREFIX="$candidate"
        break
    fi
done

if [ -n "$PIXI_GXX_PREFIX" ]; then
    PIXI_CXX_INCLUDE="$PIXI_GXX_PREFIX/include/c++"
    PIXI_CXX_TARGET_INCLUDE="$PIXI_GXX_PREFIX/include/c++/x86_64-conda-linux-gnu"
    PIXI_CXX_BACKWARD_INCLUDE="$PIXI_GXX_PREFIX/include/c++/backward"
    PIXI_GCC_INCLUDE="$PIXI_GXX_PREFIX/include"
    PIXI_GCC_FIXED_INCLUDE="$PIXI_GXX_PREFIX/include-fixed"
    PIXI_SYSROOT_INCLUDE="$PIXI_ENV_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include"

    prepend_env CPLUS_INCLUDE_PATH "$PIXI_CXX_INCLUDE:$PIXI_CXX_TARGET_INCLUDE:$PIXI_CXX_BACKWARD_INCLUDE:$PIXI_GCC_INCLUDE:$PIXI_GCC_FIXED_INCLUDE:$PIXI_SYSROOT_INCLUDE"
    prepend_env CPATH "$PIXI_GCC_INCLUDE:$PIXI_GCC_FIXED_INCLUDE:$PIXI_SYSROOT_INCLUDE"
fi
