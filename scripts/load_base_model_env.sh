#!/bin/sh

if [ -z "${REPO_ROOT:-}" ]; then
    printf '%s\n' "REPO_ROOT 未设置：请先让 direnv 自动激活 .envrc" >&2
    return 1 2>/dev/null || exit 1
fi

. "$REPO_ROOT/scripts/_require_public_env.sh"
require_public_env

PIXI_ENV_PREFIX="$REPO_ROOT/.pixi/envs/default"
PIXI_BIN="$PIXI_ENV_PREFIX/bin"

export CC="$PIXI_BIN/gcc"
export CXX="$PIXI_BIN/g++"
export CPP="$PIXI_BIN/cpp"
