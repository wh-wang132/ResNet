#!/bin/sh

if [ -z "${REPO_ROOT:-}" ]; then
    printf '%s\n' "REPO_ROOT 未设置：请先让 direnv 自动激活 .envrc" >&2
    return 1 2>/dev/null || exit 1
fi

. "$REPO_ROOT/scripts/_require_public_env.sh"
require_public_env

remove_path_entry() {
    name=$1
    target=$2
    eval "env_value=\${$name-}"

    if [ -z "$env_value" ]; then
        return 0
    fi

    cleaned=$(printf '%s' "$env_value" | awk -v RS=: -v ORS=: -v target="$target" '$0 != "" && $0 != target {print}')
    cleaned=${cleaned%:}
    eval "$name=\$cleaned"
    export "$name"
}

UV_RUNTIME_LIB_PATH=$(find "$REPO_ROOT/.venv/lib" -type d \( -path '*/site-packages/nvidia/*/lib' -o -path '*/site-packages/tensorrt_libs' \) | LC_ALL=C sort | paste -sd ':' -)

if [ -z "${UV_RUNTIME_LIB_PATH:-}" ]; then
    printf '%s\n' "未找到 UV TensorRT / NVIDIA 运行时库目录，请先确认 uv 环境完整安装" >&2
    return 1 2>/dev/null || exit 1
fi

PIXI_LIB="$REPO_ROOT/.pixi/envs/default/lib"
remove_path_entry LD_LIBRARY_PATH "$PIXI_LIB"

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="$UV_RUNTIME_LIB_PATH:$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$UV_RUNTIME_LIB_PATH"
fi

export ORT_TENSORRT_ENGINE_CACHE_ENABLE=True
export ORT_TENSORRT_CACHE_PATH="$REPO_ROOT/output/onnx_trt_cache"
export ORT_TENSORRT_ENGINE_CACHE_PATH="$ORT_TENSORRT_CACHE_PATH"
mkdir -p "$ORT_TENSORRT_CACHE_PATH"
