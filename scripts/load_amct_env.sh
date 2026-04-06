#!/bin/sh

if [ -z "${REPO_ROOT:-}" ]; then
    printf '%s\n' "REPO_ROOT 未设置：请先让 direnv 自动激活 .envrc" >&2
    return 1 2>/dev/null || exit 1
fi

. "$REPO_ROOT/scripts/_require_public_env.sh"
require_public_env

. "$REPO_ROOT/scripts/load_cann_env.sh"
