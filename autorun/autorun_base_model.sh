#!/bin/sh
set -eu

. "$REPO_ROOT/scripts/_require_public_env.sh"
require_public_env

. "$REPO_ROOT/scripts/load_base_model_env.sh"

cd "$REPO_ROOT"

uv run src/base_model_main.py --epochs 250 --batch_size 32 --model resnet6_2d --full_load True #
uv run src/base_model_main.py --epochs 250 --batch_size 64 --model resnet6_2d --full_load True
uv run src/base_model_main.py --epochs 250 --batch_size 128 --model resnet6_2d --full_load True

uv run src/base_model_main.py --epochs 200 --batch_size 32 --model resnet10_2d --full_load True #
uv run src/base_model_main.py --epochs 200 --batch_size 64 --model resnet10_2d --full_load True
uv run src/base_model_main.py --epochs 200 --batch_size 128 --model resnet10_2d --full_load True

uv run src/base_model_main.py --epochs 160 --batch_size 32 --model resnet14_2d --full_load True #
uv run src/base_model_main.py --epochs 160 --batch_size 64 --model resnet14_2d --full_load True
uv run src/base_model_main.py --epochs 160 --batch_size 128 --model resnet14_2d --full_load True

uv run src/base_model_main.py --epochs 130 --batch_size 32 --model resnet18_2d --full_load True
uv run src/base_model_main.py --epochs 130 --batch_size 64 --model resnet18_2d --full_load True
uv run src/base_model_main.py --epochs 130 --batch_size 128 --model resnet18_2d --full_load True #

uv run src/base_model_main.py --epochs 100 --batch_size 32 --model resnet34_2d --full_load True
uv run src/base_model_main.py --epochs 100 --batch_size 64 --model resnet34_2d --full_load True
uv run src/base_model_main.py --epochs 100 --batch_size 128 --model resnet34_2d --full_load True #
