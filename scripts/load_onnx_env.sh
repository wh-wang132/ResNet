#!/bin/sh

if [ -z "${REPO_ROOT:-}" ]; then
    printf '%s\n' "REPO_ROOT 未设置：请先让 direnv 自动激活 .envrc" >&2
    return 1 2>/dev/null || exit 1
fi

. "$REPO_ROOT/scripts/_require_public_env.sh"
require_public_env

PIXI_ENV_PREFIX="$REPO_ROOT/.pixi/envs/default"
PIXI_LIB="$PIXI_ENV_PREFIX/lib"

if [ -d "$PIXI_LIB" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *:"$PIXI_LIB":*) ;;
        *)
            if [ -n "${LD_LIBRARY_PATH:-}" ]; then
                export LD_LIBRARY_PATH="$PIXI_LIB:$LD_LIBRARY_PATH"
            else
                export LD_LIBRARY_PATH="$PIXI_LIB"
            fi
            ;;
    esac
fi
