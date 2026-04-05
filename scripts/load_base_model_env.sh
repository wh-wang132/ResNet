#!/bin/sh

REPO_ROOT=${REPO_ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}
PIXI_ENV_PREFIX="$REPO_ROOT/.pixi/envs/default"
PIXI_BIN="$PIXI_ENV_PREFIX/bin"

export CC="$PIXI_BIN/gcc"
export CXX="$PIXI_BIN/g++"
export CPP="$PIXI_BIN/cpp"
