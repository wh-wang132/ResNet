#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模块
包含模型测试和混淆矩阵生成功能
"""

import os
import torch
from torch.amp import autocast
from tqdm import tqdm

from confusionMatrix import ConfusionMatrix


def load_model_state_dict_safely(model, state_dict):
    """
    安全地加载模型权重，处理可能的 _orig_mod 前缀问题

    Args:
        model: 目标模型
        state_dict: 要加载的权重字典

    Returns:
        bool: 是否成功加载
    """
    try:
        # 首先尝试直接加载
        model.load_state_dict(state_dict, strict=True)
        return True
    except RuntimeError as e:
        # 如果失败，尝试处理 _orig_mod 前缀
        if "Missing key(s) in state_dict" in str(
            e
        ) and "Unexpected key(s) in state_dict" in str(e):
            print("  ⚠️  检测到权重键名不匹配，尝试处理 _orig_mod 前缀...")

            # 创建新的 state_dict，移除 _orig_mod 前缀
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("_orig_mod."):
                    new_key = key[len("_orig_mod.") :]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value

            # 尝试用处理后的 state_dict 加载
            try:
                model.load_state_dict(new_state_dict, strict=True)
                print("  ✓ 成功处理 _orig_mod 前缀并加载模型")
                return True
            except Exception as e2:
                print(f"  ✗ 处理后仍然失败: {e2}")
                return False
        else:
            print(f"  ✗ 加载失败: {e}")
            return False


def test_model(model, device, test_loader, args, folder_path, labels__):
    """
    测试模型并生成混淆矩阵

    Args:
        model: 要测试的模型
        device: 计算设备
        test_loader: 测试数据加载器
        args: 命令行参数
        folder_path: 输出目录路径
        labels__: 类别标签列表
    """
    print(f"\n{'='*80}")
    print("开始测试 (FP16)")
    print(f"{'='*80}")

    model_path = os.path.join(folder_path, args.model_path)
    assert os.path.exists(model_path), f"找不到模型文件: {model_path}"

    # 加载模型（同时兼容旧格式和新 checkpoint 格式）
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict_to_load = checkpoint["model_state_dict"]
    else:
        state_dict_to_load = checkpoint

    # 安全加载模型权重
    success = load_model_state_dict_safely(model, state_dict_to_load)
    if not success:
        raise RuntimeError(
            f"\n{'='*80}\n错误：加载模型文件时出错！\n  文件路径: {model_path}\n{'='*80}"
        )

    model.to(device)

    confusion = ConfusionMatrix(num_classes=args.class_num, labels=labels__)
    model.eval()

    with torch.no_grad(), autocast("cuda", enabled=torch.cuda.is_available()):
        for val_data in tqdm(test_loader):
            val_images, val_labels = val_data
            val_images = val_images.to(device)
            outputs = model(val_images)
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.numpy())

    confusion.plot(folder_path)
    confusion.summary()
