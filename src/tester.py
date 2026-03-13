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
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
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

