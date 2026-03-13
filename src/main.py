#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.npy 数据集 2D ResNet 训练脚本（FP16 全流程版本）
使用轻量级架构，GPU 加速，内存监控，过拟合缓解，自动混合精度 (AMP)
模块化重构版本
"""

import os
import torch
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

# 导入项目模块
from dataset import data_set_split
from utils import (
    release_gpu_memory,
    setup_device,
    parse_args,
    create_output_directory,
    load_model_map,
    print_model_info,
    create_optimized_dataloader,
)
from trainer import train_model
from tester import test_model
from visualizer import visualize_umap


def main():
    args = parse_args()
    print(args)

    # 释放 GPU 内存
    release_gpu_memory()

    # 设备配置
    device = setup_device()

    # 创建输出目录
    folder_path = create_output_directory(args)
    print(f"\n输出目录: {folder_path}")

    # 数据加载
    print("\n开始加载数据 (FP16)...")
    train_dataset, validate_dataset, test_dataset, labels__ = data_set_split(
        args.data_dir,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        full_load=args.full_load,
        num_workers=args.num_workers,
    )

    # 检查是否使用GPU
    use_cuda = torch.cuda.is_available()
    pin_memory = args.pin_memory and use_cuda

    # 优化的数据加载器配置
    train_loader, _ = create_optimized_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=pin_memory,
        drop_last=True,
        loader_name="训练集 DataLoader",
    )
    validate_loader, _ = create_optimized_dataloader(
        validate_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=pin_memory,
        drop_last=False,
        loader_name="验证集 DataLoader",
    )
    test_loader, _ = create_optimized_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=pin_memory,
        drop_last=False,
        loader_name="测试集 DataLoader",
    )

    train_num = len(train_dataset)
    val_num = len(validate_dataset)
    test_num = len(test_dataset)

    print(f"训练样本数: {train_num}, 验证样本数: {val_num}, 测试样本数: {test_num}")

    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 模型初始化
    print(f"\n初始化模型: {args.model}")
    model_map = load_model_map()
    model = model_map[args.model](num_classes=args.class_num, dropout_p=args.dropout_p)
    model.to(device)

    # 打印模型信息
    print_model_info(model, device)

    # 训练阶段
    if args.Train:
        model = train_model(
            model=model,
            device=device,
            train_loader=train_loader,
            validate_loader=validate_loader,
            args=args,
            folder_path=folder_path,
            val_num=val_num,
        )

    # 测试阶段
    if args.Test:
        test_model(
            model=model,
            device=device,
            test_loader=test_loader,
            args=args,
            folder_path=folder_path,
            labels__=labels__,
        )

    # UMAP 可视化
    if args.UMAP:
        visualize_umap(
            model=model,
            device=device,
            test_loader=test_loader,
            args=args,
            folder_path=folder_path,
            labels__=labels__,
        )

    print(f"\n{'='*80}")
    print("所有任务完成")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
