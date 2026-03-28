#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""剪枝阶段参数解析。"""

import argparse

from .utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser(
        description="Structured pruning + finetuning pipeline based on torch-pruning"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "resnet6_2d",
            "resnet10_2d",
            "resnet14_2d",
            "resnet18_2d",
            "resnet34_2d",
            "resnet50_2d",
        ],
        help="基座模型名，将自动解析 output/base_model/<model>/best_model.pth 符号链接",
    )
    parser.add_argument("--model_path", type=str, default="best_pruned_model.pth", help="剪枝后最佳模型保存文件名")
    parser.add_argument("--data_dir", type=str, default="Data", help="数据集路径")
    parser.add_argument("--class_num", type=int, default=24, help="分类数")
    parser.add_argument(
        "--data_dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="数据加载后的 tensor 精度，仅影响数据集输出",
    )
    parser.add_argument("--full_load", type=str2bool, default=False, help="是否全量加载数据集")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader 工作线程数")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader 预取因子")
    parser.add_argument("--persistent_workers", type=str2bool, default=True, help="是否保持 DataLoader 工作线程")
    parser.add_argument("--pin_memory", type=str2bool, default=True, help="是否启用 pin_memory")

    parser.add_argument("--pruning_ratio", type=float, default=0.3, help="iterative pruning 的最终总剪枝率")
    parser.add_argument("--pruning_steps", type=int, default=5, help="多轮 iterative pruning 的剪枝轮数")
    parser.add_argument("--global_pruning", type=str2bool, default=True, help="是否启用全局剪枝")
    parser.add_argument("--ignore_fc", type=str2bool, default=True, help="是否默认忽略分类头")

    parser.add_argument("--finetune_epochs", type=int, default=10, help="每轮剪枝后的微调轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="微调学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup 占总步数比例")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup 步数，0 表示使用 warmup_ratio")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="最小学习率")
    parser.add_argument("--cudnn_benchmark", type=str2bool, default=True, help="是否启用 cuDNN benchmark")
    parser.add_argument("--cudnn_deterministic", type=str2bool, default=False, help="是否启用 cuDNN 确定性算法")
    parser.add_argument("--evaluate_test", type=str2bool, default=True, help="微调结束后是否评估测试集")

    return parser.parse_args()
