#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用辅助函数模块
包含 GPU 内存管理、配置和打印辅助函数
"""

import os
import gc
import hashlib
import torch
import argparse
from torch.amp import autocast


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


def build_architecture_signature(model):
    """
    基于 state_dict 形状信息生成结构签名，用于跨阶段一致性校验
    """
    state_dict = model.state_dict()
    shape_items = []
    for key, value in state_dict.items():
        shape = list(value.shape)
        shape_items.append((key, shape))

    # 以键名排序保证签名稳定
    shape_items.sort(key=lambda x: x[0])
    canonical_text = "|".join(
        f"{key}:{','.join(map(str, shape))}" for key, shape in shape_items
    )
    signature_hash = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()

    return {
        "signature_algo": "sha256",
        "signature_hash": signature_hash,
        "state_dict_shapes": {key: shape for key, shape in shape_items},
        "parameter_count": int(sum(param.numel() for param in model.parameters())),
    }


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
    output_path = "./output/base_model"
    folder_path = os.path.join(
        output_path,
        args.model,
        f"epochs{args.epochs}_bs{args.batch_size}",
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
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load_model_map():
    """加载模型映射"""
    from .resnet_lightweight import resnet6_2d, resnet10_2d, resnet14_2d
    from .resnet_standard import resnet18_2d, resnet34_2d, resnet50_2d

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
        with autocast("cuda", enabled=True):
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


def remove_orig_mod_prefix(state_dict):
    """
    移除权重字典中的 _orig_mod 前缀

    Args:
        state_dict: 原始权重字典

    Returns:
        dict: 处理后的权重字典
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_k = k[len("_orig_mod.") :]
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_raw_model(model):
    """
    获取编译模型的原始模型（如果有的话）

    Args:
        model: 可能被torch.compile包装的模型

    Returns:
        nn.Module: 原始模型
    """
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    if hasattr(model, "module"):
        return model.module
    return model


def load_state_dict_safely(model, state_dict, strict=True):
    """
    安全地加载模型权重，智能处理 _orig_mod 前缀问题（只移除前缀，不添加）

    Args:
        model: 目标模型
        state_dict: 要加载的权重字典
        strict: 是否严格匹配键名

    Returns:
        bool: 是否成功加载
    """

    try:
        # 首先尝试直接加载到原始模型（如果是编译模型）
        raw_model = get_raw_model(model)
        raw_model.load_state_dict(state_dict, strict=strict)
        return True
    except RuntimeError as e:
        error_str = str(e)
        if (
            "Missing key(s) in state_dict" in error_str
            and "Unexpected key(s) in state_dict" in error_str
        ):
            # 检测到键名不匹配，尝试处理
            has_orig_mod_in_state = any(
                k.startswith("_orig_mod.") for k in state_dict.keys()
            )

            if has_orig_mod_in_state:
                # 状态字典有前缀，尝试移除前缀后加载
                new_state_dict = remove_orig_mod_prefix(state_dict)
                try:
                    raw_model = get_raw_model(model)
                    raw_model.load_state_dict(new_state_dict, strict=strict)
                    return True
                except Exception:
                    pass

            # 最后尝试非严格加载
            if not strict:
                try:
                    raw_model = get_raw_model(model)
                    raw_model.load_state_dict(state_dict, strict=False)
                    return True
                except Exception:
                    pass

        return False
