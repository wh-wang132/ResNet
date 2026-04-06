#!/bin/sh

fail_public_env() {
    printf '%s\n' "$1" >&2
    return 1 2>/dev/null || exit 1
}

require_public_env() {
    if [ -z "${REPO_ROOT:-}" ]; then
        fail_public_env "REPO_ROOT 未设置：请先进入项目目录并让 direnv 自动激活 .envrc"
    fi

    if [ ! -f "$REPO_ROOT/.envrc" ] || [ ! -d "$REPO_ROOT/src" ]; then
        fail_public_env "REPO_ROOT 无效：当前公共环境未指向合法仓库根目录"
    fi

    expected_pythonpath="$REPO_ROOT/src"
    case ":${PYTHONPATH:-}:" in
        *:"$expected_pythonpath":*) ;;
        *)
            fail_public_env \
                "PYTHONPATH 缺少 $expected_pythonpath：请先让 direnv 自动激活 .envrc"
            ;;
    esac
}
