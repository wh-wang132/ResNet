#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用辅助函数模块
包含 GPU 内存管理、配置和打印辅助函数
"""

import os
import gc
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter


def release_gpu_memory():
    """释放 GPU 内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_gpu_memory_info():
    """获取 GPU 内存使用信息"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        cached = torch.cuda.memory_reserved() / (1024**3)
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        return {"allocated_gb": allocated, "cached_gb": cached, "peak_gb": peak}
    return {"allocated_gb": 0, "cached_gb": 0, "peak_gb": 0}


def print_training_summary(table_title, train_loss, val_loss, val_acc, gpu_info, epoch):
    """打印训练摘要"""
    print(f"\n{'='*80}")
    print(f"{table_title} (Epoch {epoch+1})")
    print(f"{'='*80}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Acc: {val_acc:.4f}")
    print(
        f"GPU Memory: Allocated {gpu_info['allocated_gb']:.2f} GB, Cached {gpu_info['cached_gb']:.2f} GB"
    )
    print(f"Peak Memory: {gpu_info['peak_gb']:.2f} GB")


def create_output_directory(args):
    """创建输出目录"""
    folder_name = (
        os.path.basename(args.data_dir)
        if os.path.basename(args.data_dir)
        else "npy_dataset"
    )
    output_path = "./output"
    folder_path = os.path.join(
        output_path,
        f"{args.model}_{folder_name}",
        f"epochs{args.epochs}_bs{args.batch_size}_lr{args.lr}_drop{args.dropout_p}",
    )
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


def setup_device():
    """设置计算设备"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )
    return device


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Train a 2D Lightweight ResNet for .npy Dataset"
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数 (默认 60)")
    parser.add_argument("--lr", type=float, default=0.0003, help="学习率 (默认 0.001)")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小 (默认 32)")
    parser.add_argument(
        "--model_path", type=str, default="best_model.pth", help="模型保存路径"
    )
    parser.add_argument("--class_num", type=int, default=24, help="分类数 (默认 24)")

    # 模型选择
    parser.add_argument(
        "--model",
        type=str,
        default="resnet6_2d",
        choices=[
            "resnet6_2d",
            "resnet10_2d",
            "resnet14_2d",
            "resnet18_2d",
            "resnet34_2d",
            "resnet50_2d",
        ],
        help="选择模型 (默认 resnet6_2d)",
    )

    # 数据路径
    parser.add_argument("--data_dir", type=str, default="Data", help="数据集路径")

    # 数据加载选项
    parser.add_argument(
        "--full_load",
        action="store_true",
        default=False,
        help="全量加载数据集到内存 (默认 False)",
    )
    parser.add_argument(
        "--no-full_load",
        dest="full_load",
        action="store_false",
        help="禁用全量加载（增量加载）",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="数据加载工作线程数 (None=自动检测CPU核心数)",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="DataLoader预取因子 (默认 2)",
    )
    parser.add_argument(
        "--persistent_workers",
        action="store_true",
        default=True,
        help="保持DataLoader工作线程活跃 (默认 True)",
    )
    parser.add_argument(
        "--no-persistent_workers",
        dest="persistent_workers",
        action="store_false",
        help="禁用持久化工作线程",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=True,
        help="启用CUDA内存钉住 (默认 True, GPU训练时启用)",
    )
    parser.add_argument(
        "--no-pin_memory",
        dest="pin_memory",
        action="store_false",
        help="禁用内存钉住",
    )

    # 功能开关
    parser.add_argument(
        "--Train", action="store_true", default=True, help="是否训练 (默认 True)"
    )
    parser.add_argument(
        "--no-Train", dest="Train", action="store_false", help="禁用训练"
    )
    parser.add_argument(
        "--Test", action="store_true", default=True, help="是否测试 (默认 True)"
    )
    parser.add_argument("--no-Test", dest="Test", action="store_false", help="禁用测试")
    parser.add_argument(
        "--UMAP", action="store_true", default=False, help="UMAP 可视化 (默认 False)"
    )

    # 正则化参数
    parser.add_argument(
        "--dropout_p", type=float, default=0.3, help="Dropout 概率 (默认 0.3)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="权重衰减 (默认 1e-4)"
    )

    # 学习率调度器参数
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Warmup 占总步数的比例 (默认 0.05, 即 5%%)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Warmup 步数 (如果为 0，则使用 warmup_ratio, 默认 0)",
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-6, help="最小学习率 (默认 1e-6)"
    )
    parser.add_argument(
        "--plot_lr_schedule",
        action="store_true",
        default=True,
        help="是否绘制学习率调度曲线 (默认 True)",
    )
    parser.add_argument(
        "--no-plot_lr_schedule",
        dest="plot_lr_schedule",
        action="store_false",
        help="禁用学习率调度曲线绘图",
    )

    return parser.parse_args()


def load_model_map():
    """加载模型映射"""
    from resnet_lightweight import resnet6_2d, resnet10_2d, resnet14_2d
    from resnet_standard import resnet18_2d, resnet34_2d, resnet50_2d

    return {
        "resnet6_2d": resnet6_2d,
        "resnet10_2d": resnet10_2d,
        "resnet14_2d": resnet14_2d,
        "resnet18_2d": resnet18_2d,
        "resnet34_2d": resnet34_2d,
        "resnet50_2d": resnet50_2d,
    }


def print_model_info(model, device):
    """打印模型信息"""
    try:
        from torchsummary import summary

        print("\n模型结构:")
        summary(model, input_size=(1, 543, 512), device=str(device).split(":")[0])
    except Exception as e:
        print(f"无法使用 torchsummary: {e}")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")


def create_optimized_dataloader(
    dataset,
    batch_size,
    shuffle=False,
    num_workers=None,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    loader_name="DataLoader",
):
    """
    创建优化的DataLoader，带有详细日志

    Args:
        dataset: PyTorch Dataset实例
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作线程数（None=自动检测）
        prefetch_factor: 预取因子
        persistent_workers: 保持工作线程活跃
        pin_memory: 内存钉住（GPU训练时推荐）
        drop_last: 是否丢弃最后一个不完整批次
        loader_name: 加载器名称（用于日志）

    Returns:
        DataLoader实例
    """
    import torch

    # 自动计算工作线程数
    if num_workers is None:
        cpu_count = os.cpu_count() or 4
        num_workers = min(cpu_count, batch_size if batch_size > 1 else 0, 8)
        num_workers = max(num_workers, 0)

    # 打印配置信息
    print(f"\n{'='*80}")
    print(f"配置 {loader_name}:")
    print(f"{'='*80}")
    print(f"  批次大小: {batch_size}")
    print(f"  打乱数据: {shuffle}")
    print(f"  工作线程数: {num_workers}")
    print(f"  预取因子: {prefetch_factor}")
    print(f"  持久化工作线程: {persistent_workers}")
    print(f"  内存钉住: {pin_memory}")
    print(f"  丢弃最后批次: {drop_last}")
    print(f"{'='*80}\n")

    # 只有当num_workers > 0时，prefetch_factor和persistent_workers才有效
    if num_workers == 0:
        prefetch_factor = None
        persistent_workers = False

    # 创建DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataloader, num_workers
