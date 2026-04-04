#!/bin/sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

set -eu

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
    checkpoint_name=$3
    tmp_list=$(mktemp)
    tmp_files="$tmp_files $tmp_list"

    print_section "开始遍历 ${branch}: ${root_dir}"

    find "$root_dir" -type f -name "$checkpoint_name" | LC_ALL=C sort > "$tmp_list"

    while IFS= read -r checkpoint_path; do
        [ -n "$checkpoint_path" ] || continue
        printf '\n[%s] %s\n' "$branch" "$checkpoint_path"
        uv run src/onnx_main.py \
            --branch "$branch" \
            --checkpoint "$checkpoint_path" \
            # --full_load True
    done < "$tmp_list"

    print_section "完成遍历 ${branch}: ${root_dir}"
}

run_branch "pruning_fp16" "output/pruning" "best_pruned_model.pth"
run_branch "qat_convert" "output/qat" "best_qat_prepare_model.pth"
