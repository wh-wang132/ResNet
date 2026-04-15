#!/bin/sh
set -eu

. "$REPO_ROOT/scripts/_require_public_env.sh"
require_public_env

. "$REPO_ROOT/scripts/load_amct_env.sh"

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

tmp_list=$(mktemp)
tmp_files="$tmp_files $tmp_list"

cd "$REPO_ROOT"

print_section "开始遍历 qat_convert ONNX -> AMCT"
find "output/onnx/qat_convert" -type f -name "model_quant.onnx" | LC_ALL=C sort > "$tmp_list"

while IFS= read -r onnx_model_path; do
    [ -n "$onnx_model_path" ] || continue
    printf '\n[amct] %s\n' "$onnx_model_path"
    uv run python -m amct --onnx_model "$onnx_model_path"
done < "$tmp_list"

print_section "完成遍历 qat_convert ONNX -> AMCT"
