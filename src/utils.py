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


def str2bool(v):
    """
    正确的字符串到布尔值转换器

    Args:
        v: 字符串值

    Returns:
        bool: 解析后的布尔值

    Raises:
        argparse.ArgumentTypeError: 如果无法解析为布尔值
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        import argparse

        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Train a 2D Lightweight ResNet for .npy Dataset"
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=60, help="训练轮数 (默认 60)")
    parser.add_argument("--lr", type=float, default=0.0003, help="学习率 (默认 0.003)")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小 (默认 64)")
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
        type=str2bool,
        default=False,
        help="全量加载数据集到内存 (默认 False)",
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
        type=str2bool,
        default=True,
        help="保持DataLoader工作线程活跃 (默认 True)",
    )
    parser.add_argument(
        "--pin_memory",
        type=str2bool,
        default=True,
        help="启用CUDA内存钉住 (默认 True, GPU训练时启用)",
    )

    # cuDNN和性能优化选项
    parser.add_argument(
        "--cudnn_benchmark",
        type=str2bool,
        default=True,
        help="启用cuDNN自动调优 (默认 True, 优化卷积性能)",
    )
    parser.add_argument(
        "--cudnn_deterministic",
        type=str2bool,
        default=False,
        help="启用cuDNN确定性算法 (默认 False, 禁用以提高速度)",
    )
    parser.add_argument(
        "--compile_model",
        type=str2bool,
        default=True,
        help="启用模型编译 (默认 True, 使用torch.compile优化)",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        help="模型编译模式 (默认 default)",
    )

    # 功能开关
    parser.add_argument(
        "--Train",
        type=str2bool,
        default=True,
        help="是否训练 (默认 True)",
    )
    parser.add_argument(
        "--Test",
        type=str2bool,
        default=True,
        help="是否测试 (默认 True)",
    )
    parser.add_argument(
        "--UMAP",
        type=str2bool,
        default=False,
        help="UMAP 可视化 (默认 False)",
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
        type=str2bool,
        default=True,
        help="是否绘制学习率调度曲线 (默认 True)",
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


def configure_cudnn(args):
    """
    配置cuDNN设置并验证配置是否生效

    Args:
        args: 命令行参数

    Returns:
        dict: 包含配置状态的字典
    """
    print(f"\n{'='*80}")
    print("配置 cuDNN 性能优化")
    print(f"{'='*80}")

    config_status = {
        "cudnn_available": False,
        "cudnn_benchmark": args.cudnn_benchmark,
        "cudnn_deterministic": args.cudnn_deterministic,
        "cudnn_version": None,
    }

    if torch.cuda.is_available():
        config_status["cudnn_available"] = torch.backends.cudnn.is_available()

        if config_status["cudnn_available"]:
            config_status["cudnn_version"] = torch.backends.cudnn.version()

            # 设置cuDNN benchmark模式（自动调优）
            torch.backends.cudnn.benchmark = args.cudnn_benchmark
            print(f"  cuDNN 自动调优: {'✓ 启用' if args.cudnn_benchmark else '✗ 禁用'}")

            # 设置确定性算法
            torch.backends.cudnn.deterministic = args.cudnn_deterministic

            if args.cudnn_deterministic:
                print(f"  cuDNN 确定性算法: ✓ 启用 (可能降低性能)")
                # 当使用确定性算法时，禁用benchmark
                if args.cudnn_benchmark:
                    print("  ⚠️  警告: 确定性算法已启用，自动禁用 cuDNN benchmark")
                    torch.backends.cudnn.benchmark = False
            else:
                print(f"  cuDNN 非确定性算法: ✓ 启用 (提高训练速度)")

            print(f"  cuDNN 版本: {config_status['cudnn_version']}")

            # 验证配置是否生效
            actual_benchmark = torch.backends.cudnn.benchmark
            actual_deterministic = torch.backends.cudnn.deterministic

            if actual_benchmark == args.cudnn_benchmark or (
                args.cudnn_deterministic and not actual_benchmark
            ):
                print(f"  ✓ cuDNN benchmark 配置验证通过")
            else:
                print(f"  ✗ cuDNN benchmark 配置验证失败")

            if actual_deterministic == args.cudnn_deterministic:
                print(f"  ✓ cuDNN 确定性算法配置验证通过")
            else:
                print(f"  ✗ cuDNN 确定性算法配置验证失败")
        else:
            print("  ✗ cuDNN 不可用")
    else:
        print("  ℹ️  CUDA 不可用，跳过 cuDNN 配置")

    print(f"{'='*80}\n")
    return config_status


def compile_model(model, args, device, loss_function, optimizer):
    """
    编译模型以优化性能

    Args:
        model: 要编译的模型
        args: 命令行参数
        device: 计算设备
        loss_function: 损失函数（用于编译验证）
        optimizer: 优化器（用于编译验证）

    Returns:
        compiled_model: 编译后的模型（如果禁用编译则返回原模型）
    """
    if not args.compile_model or not torch.cuda.is_available():
        if not args.compile_model:
            print("\nℹ️  模型编译已禁用")
        else:
            print("\nℹ️  CUDA 不可用，跳过模型编译")
        return model

    print(f"\n{'='*80}")
    print("开始模型编译")
    print(f"{'='*80}")
    print(f"  编译模式: {args.compile_mode}")
    print(f"  正在编译模型，请稍候...")

    import time

    start_time = time.time()

    try:
        # 使用torch.compile编译模型
        compiled_model = torch.compile(
            model,
            mode=args.compile_mode,
            fullgraph=False,
            dynamic=False,
        )

        # 验证编译是否成功 - 通过一次前向+反向传播
        compiled_model.train()

        # 创建一个示例输入进行验证
        sample_input = torch.randn(1, 1, 543, 512, dtype=torch.float16, device=device)
        sample_target = torch.randint(0, 24, (1,), device=device)

        # 执行前向传播
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=True):
            sample_output = compiled_model(sample_input)
            sample_loss = loss_function(sample_output, sample_target)

        # 执行反向传播
        sample_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        compile_time = time.time() - start_time

        print(f"  ✓ 模型编译成功")
        print(f"  编译耗时: {compile_time:.2f}s")
        print(f"  编译模式: {args.compile_mode}")
        print(f"{'='*80}\n")

        return compiled_model

    except Exception as e:
        compile_time = time.time() - start_time
        print(f"  ✗ 模型编译失败: {e}")
        print(f"  编译耗时: {compile_time:.2f}s")
        print(f"  ⚠️  回退到未编译模型")
        print(f"{'='*80}\n")

        # 重置优化器状态
        optimizer.zero_grad()

        return model


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
