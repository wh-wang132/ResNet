#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QAT 阶段参数解析。"""

import argparse

from .utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conservative torch native FX QAT pipeline from pruning checkpoint"
    )

    parser.add_argument(
        "--pruning_checkpoint",
        type=str,
        required=True,
        help="输入 pruning checkpoint 路径",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="best_qat_prepare_model.pth",
        help="QAT prepare 模型保存文件名",
    )
    parser.add_argument("--data_dir", type=str, default="Data", help="数据集路径")
    parser.add_argument(
        "--data_dtype",
        type=str,
        default="fp32",
        choices=["fp32"],
        help="数据加载后的 tensor 精度（QAT 固定为 fp32）",
    )
    parser.add_argument("--full_load", type=str2bool, default=False, help="是否全量加载数据集")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader 工作线程数")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader 预取因子")
    parser.add_argument(
        "--persistent_workers",
        type=str2bool,
        default=True,
        help="是否保持 DataLoader 工作线程",
    )
    parser.add_argument("--pin_memory", type=str2bool, default=True, help="是否启用 pin_memory")

    parser.add_argument("--qat_epochs", type=int, default=10, help="QAT 微调轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-5, help="QAT 微调学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup 占总步数比例")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup 步数，0 表示使用 warmup_ratio")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="最小学习率")
    parser.add_argument("--cudnn_benchmark", type=str2bool, default=True, help="是否启用 cuDNN benchmark")
    parser.add_argument(
        "--cudnn_deterministic",
        type=str2bool,
        default=False,
        help="是否启用 cuDNN 确定性算法",
    )
    parser.add_argument("--evaluate_test", type=str2bool, default=True, help="QAT 结束后是否评估测试集")

    return parser.parse_args()
