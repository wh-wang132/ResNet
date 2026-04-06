#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/scripts/_require_public_env.sh"
require_public_env

cd "$REPO_ROOT"

uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet6_2d/ratio0.30_steps2_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet6_2d/ratio0.40_steps5_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet6_2d/ratio0.50_steps5_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet6_2d/ratio0.50_steps8_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet6_2d/ratio0.50_steps12_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet6_2d/ratio0.55_steps12_global_ft10_bs64/best_pruned_model.pth --full_load True #
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet6_2d/ratio0.55_steps16_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet6_2d/ratio0.60_steps5_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet6_2d/ratio0.60_steps8_global_ft10_bs64/best_pruned_model.pth --full_load True

uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet10_2d/ratio0.40_steps5_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet10_2d/ratio0.50_steps5_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet10_2d/ratio0.60_steps5_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet10_2d/ratio0.60_steps8_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet10_2d/ratio0.60_steps12_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet10_2d/ratio0.60_steps16_global_ft10_bs64/best_pruned_model.pth --full_load True #
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet10_2d/ratio0.65_steps12_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet10_2d/ratio0.70_steps8_global_ft10_bs64/best_pruned_model.pth --full_load True

uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet14_2d/ratio0.50_steps5_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet14_2d/ratio0.60_steps5_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet14_2d/ratio0.60_steps8_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet14_2d/ratio0.70_steps8_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet14_2d/ratio0.70_steps12_global_ft10_bs64/best_pruned_model.pth --full_load True #
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet14_2d/ratio0.70_steps16_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet14_2d/ratio0.80_steps8_global_ft10_bs64/best_pruned_model.pth --full_load True

uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet18_2d/ratio0.50_steps5_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet18_2d/ratio0.60_steps5_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet18_2d/ratio0.60_steps8_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet18_2d/ratio0.70_steps8_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet18_2d/ratio0.80_steps8_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet18_2d/ratio0.80_steps12_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet18_2d/ratio0.80_steps16_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet18_2d/ratio0.85_steps16_global_ft10_bs64/best_pruned_model.pth --full_load True #
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet18_2d/ratio0.90_steps8_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet18_2d/ratio0.90_steps12_global_ft10_bs64/best_pruned_model.pth --full_load True

uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet34_2d/ratio0.60_steps5_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet34_2d/ratio0.60_steps8_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet34_2d/ratio0.70_steps8_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet34_2d/ratio0.80_steps8_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet34_2d/ratio0.90_steps8_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet34_2d/ratio0.90_steps12_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet34_2d/ratio0.92_steps12_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet34_2d/ratio0.92_steps20_global_ft10_bs64/best_pruned_model.pth --full_load True
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet34_2d/ratio0.94_steps20_global_ft10_bs64/best_pruned_model.pth --full_load True #
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet34_2d/ratio0.95_steps20_global_ft10_bs64/best_pruned_model.pth --full_load True
