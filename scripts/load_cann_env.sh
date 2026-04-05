#!/bin/sh

REPO_ROOT=${REPO_ROOT:-$(pwd)}
PIXI_ENV_PREFIX="$REPO_ROOT/.pixi/envs/default"
PIXI_BIN="$PIXI_ENV_PREFIX/bin"
PIXI_LIB="$PIXI_ENV_PREFIX/lib"
CANN_ROOT="$REPO_ROOT/.pixi/envs/default/Ascend/cann-8.5.0"

export REPO_ROOT
export PIXI_ENV_PREFIX
export PIXI_BIN
export PIXI_LIB
export CANN_ROOT

if [ ! -d "$CANN_ROOT" ]; then
    echo "CANN toolkit not found: $CANN_ROOT" >&2
    return 1 2>/dev/null || exit 1
fi

TEST_DATA_ROOT_PATH=${TEST_DATA_ROOT_PATH:-/tmp/tvm_test_data}
mkdir -p "$TEST_DATA_ROOT_PATH"
export TEST_DATA_ROOT_PATH

append_env() {
    name=$1
    value=$2
    eval "env_value=\${$name-}"

    if [ -z "$env_value" ]; then
        eval "$name=\$value"
    else
        eval "$name=\$env_value:\$value"
    fi
    export "$name"
}

prepend_env() {
    name=$1
    value=$2
    eval "env_value=\${$name-}"

    if [ -z "$env_value" ]; then
        eval "$name=\$value"
    else
        eval "$name=\$value:\$env_value"
    fi
    export "$name"
}

remove_env() {
    name=$1
    regex=$2
    eval "env_value=\${$name-}"

    if [ -z "$env_value" ]; then
        return 0
    fi

    cleaned=$(printf '%s' "$env_value" | tr ':' '\n' | grep -v -E "$regex" | paste -sd ':' -)
    eval "$name=\$cleaned"
    export "$name"
}

has_ascend_driver() {
    dep_hal_name="libascend_hal.so"

    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
        old_ifs=$IFS
        IFS=:
        for path in $LD_LIBRARY_PATH; do
            IFS=$old_ifs
            [ -d "$path" ] || continue
            case "$path" in
                *driver*)
                    if find "$path" -name "$dep_hal_name" -print -quit 2>/dev/null | grep -q .; then
                        return 0
                    fi
                    ;;
            esac
        done
        IFS=$old_ifs
    fi

    if [ -f /etc/ascend_install.info ]; then
        driver_install_path_param=$(grep -iw driver_install_path_param /etc/ascend_install.info | cut --only-delimited -d '=' -f2-)
        if [ -n "$driver_install_path_param" ]; then
            driver_path="$driver_install_path_param/driver/lib64"
            if [ -d "$driver_path" ] && find "$driver_path" -name "$dep_hal_name" -print -quit 2>/dev/null | grep -q .; then
                return 0
            fi
        fi
    fi

    if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q -- "$dep_hal_name"; then
        return 0
    fi

    return 1
}

if command -v arch >/dev/null 2>&1; then
    ARCHITECTURE=$(arch)
else
    ARCHITECTURE=$(uname -m)
fi

INSTALL_ROOT=$(dirname "$CANN_ROOT")
REMOVE_REGEX="^${INSTALL_ROOT}/cann[/_-]"
LD_REMOVE_REGEX="^${INSTALL_ROOT}/cann[/_-]|^${PIXI_LIB}$|^/usr/local/Ascend/driver/lib64(|/common|/driver)$"
remove_env PATH "$REMOVE_REGEX"
remove_env LD_LIBRARY_PATH "$LD_REMOVE_REGEX"
remove_env CMAKE_PREFIX_PATH "$REMOVE_REGEX"

case ":${PATH:-}:" in
    *:/sbin:*) ;;
    *) PATH="${PATH:-}:/sbin"; export PATH ;;
esac

prepend_env PATH "$CANN_ROOT/bin:$CANN_ROOT/tools/ccec_compiler/bin:$CANN_ROOT/tools/profiler/bin:$CANN_ROOT/tools/ascend_system_advisor/asys:$CANN_ROOT/tools/show_kernel_debug_data:$CANN_ROOT/tools/msobjdump"
prepend_env LD_LIBRARY_PATH "$CANN_ROOT/lib64:$CANN_ROOT/lib64/plugin/opskernel:$CANN_ROOT/lib64/plugin/nnengine:$CANN_ROOT/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/$ARCHITECTURE:$CANN_ROOT/tools/aml/lib64:$CANN_ROOT/tools/aml/lib64/plugin:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
prepend_env CMAKE_PREFIX_PATH "$CANN_ROOT/lib64/cmake"
prepend_env CMAKE_PREFIX_PATH "$CANN_ROOT/toolkit/tools/tikicpulib/lib/cmake"
if [ -d "$PIXI_LIB" ]; then
    prepend_env LD_LIBRARY_PATH "$PIXI_LIB"
fi

if ! has_ascend_driver; then
    append_env LD_LIBRARY_PATH "$CANN_ROOT/devlib"
fi

export ASCEND_OPP_PATH="$CANN_ROOT/opp"
export ASCEND_AICPU_PATH="$CANN_ROOT"
export TOOLCHAIN_HOME="$CANN_ROOT/toolkit"
export ASCEND_HOME_PATH="$CANN_ROOT"
export ASCEND_TOOLKIT_HOME="$CANN_ROOT"
