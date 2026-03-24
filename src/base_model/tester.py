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

from .confusionMatrix import ConfusionMatrix
from .utils import load_state_dict_safely


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
    success = load_state_dict_safely(model, state_dict_to_load, strict=True)
    if not success:
        # 尝试非严格加载
        success = load_state_dict_safely(model, state_dict_to_load, strict=False)
        if success:
            print("  ⚠️  使用非严格模式加载模型")
        else:
            raise RuntimeError(
                f"\n{'='*80}\n错误：加载模型文件时出错！\n  文件路径: {model_path}\n{'='*80}"
            )
    else:
        print("  ✓ 成功加载模型权重")

    model.to(device)

    confusion = ConfusionMatrix(num_classes=args.class_num, labels=labels__)
    model.eval()

    with torch.no_grad(), autocast("cuda", enabled=torch.cuda.is_available()):
        for test_data in tqdm(test_loader):
            test_images, test_labels = test_data
            test_images = test_images.to(device)
            logits = model(test_images)
            # 评估后处理保持在测试流程中，不进入模型导出图
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            confusion.update(predictions.to("cpu").numpy(), test_labels.numpy())

    confusion.plot(folder_path)
    confusion.summary()
