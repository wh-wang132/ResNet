#!/bin/sh

if [ -z "${REPO_ROOT:-}" ]; then
    printf '%s\n' "REPO_ROOT 未设置：请先让 direnv 自动激活 .envrc" >&2
    return 1 2>/dev/null || exit 1
fi

. "$REPO_ROOT/scripts/_require_public_env.sh"
require_public_env

PIXI_GXX_PREFIX=""

. "$REPO_ROOT/scripts/load_cann_env.sh"

if [ -d "$PIXI_BIN" ]; then
    case ":${PATH:-}:" in
        *:"$PIXI_BIN":*) ;;
        *) export PATH="$PIXI_BIN:$PATH" ;;
    esac
fi

remove_env PYTHONPATH "^${CANN_ROOT}/python/site-packages$|^${CANN_ROOT}/opp/built-in/op_impl/ai_core/tbe$"
prepend_env PYTHONPATH "$CANN_ROOT/python/site-packages:$CANN_ROOT/opp/built-in/op_impl/ai_core/tbe"

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

    remove_env CPLUS_INCLUDE_PATH "^${PIXI_GXX_PREFIX}/include/c\\+\\+|^${PIXI_GCC_INCLUDE}$|^${PIXI_GCC_FIXED_INCLUDE}$|^${PIXI_SYSROOT_INCLUDE}$"
    remove_env CPATH "^${PIXI_GCC_INCLUDE}$|^${PIXI_GCC_FIXED_INCLUDE}$|^${PIXI_SYSROOT_INCLUDE}$"
    prepend_env CPLUS_INCLUDE_PATH "$PIXI_CXX_INCLUDE:$PIXI_CXX_TARGET_INCLUDE:$PIXI_CXX_BACKWARD_INCLUDE:$PIXI_GCC_INCLUDE:$PIXI_GCC_FIXED_INCLUDE:$PIXI_SYSROOT_INCLUDE"
    prepend_env CPATH "$PIXI_GCC_INCLUDE:$PIXI_GCC_FIXED_INCLUDE:$PIXI_SYSROOT_INCLUDE"
fi
