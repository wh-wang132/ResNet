#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/scripts/_require_public_env.sh"
require_public_env

. "$REPO_ROOT/scripts/load_atc_env.sh"

tmp_files=""

cleanup() {
    for tmp_file in $tmp_files; do
        [ -n "$tmp_file" ] && [ -f "$tmp_file" ] && rm -f "$tmp_file"
    done
}

trap cleanup EXIT HUP INT TERM

print_section() {
    title=$1
    printf '\n============================================================\n'
    printf '%s\n' "$title"
    printf '============================================================\n'
}

run_branch() {
    branch=$1
    root_dir=$2
    model_name=$3
    tmp_list=$(mktemp)
    tmp_files="$tmp_files $tmp_list"

    print_section "开始遍历 ${branch}: ${root_dir}"

    find "$root_dir" -type f -name "$model_name" | LC_ALL=C sort > "$tmp_list"

    while IFS= read -r onnx_model_path; do
        [ -n "$onnx_model_path" ] || continue
        printf '\n[%s] %s\n' "$branch" "$onnx_model_path"
        pixi run python src/atc_main.py \
            --branch "$branch" \
            --onnx_model "$onnx_model_path"
    done < "$tmp_list"

    print_section "完成遍历 ${branch}: ${root_dir}"
}

cd "$REPO_ROOT"

run_branch "pruning_fp16" "output/onnx/pruning_fp16" "model_fp16.onnx"
run_branch "amct_deploy" "output/amct" "deploy_model.onnx"
