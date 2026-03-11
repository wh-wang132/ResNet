#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.npy 数据集加载模块
用于加载 Data 目录下的 24 类.npy 格式数据集
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms


class NPYDataset(Dataset):
    """自定义 Dataset 类用于加载.npy 文件（FP16 版本）"""

    def __init__(self, file_paths, labels, transform=None):
        """
        Args:
            file_paths: .npy 文件路径列表
            labels: 对应的标签列表
            transform: 数据变换
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """加载单个样本（FP16）"""
        try:
            # 加载.npy 文件
            data = np.load(self.file_paths[idx])
            label = self.labels[idx]

            # 确保数据类型为 float16
            if data.dtype != np.float16:
                data = data.astype(np.float16)

            # 转换为 torch tensor (保持 float16)
            data = torch.from_numpy(data).to(torch.float16)

            # 添加通道维度 (H, W) -> (1, H, W)
            data = data.unsqueeze(0)

            # 应用变换
            if self.transform:
                data = self.transform(data)

            return data, label

        except Exception as e:
            print(f"Error loading {self.file_paths[idx]}: {e}")
            # 返回空样本（为了训练继续）
            return torch.zeros(1, 543, 512, dtype=torch.float16), 0


def data_set_split(
    data_dir,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    random_state=42,
):
    """
    划分数据集（FP16 版本）

    Args:
        data_dir: 数据根目录（如./Data）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子

    Returns:
        train_dataset, validate_dataset, test_dataset, labels__
    """
    file_paths = []
    labels = []
    labels__ = []
    label_map = {}
    label_index = 0

    # 读取所有文件路径和标签
    for label_folder in sorted(os.listdir(data_dir)):
        label_folder_path = os.path.join(data_dir, label_folder)
        if os.path.isdir(label_folder_path):
            labels__.append(label_folder)
            for file_name in os.listdir(label_folder_path):
                if file_name.endswith(".npy"):
                    file_path = os.path.join(label_folder_path, file_name)
                    file_paths.append(file_path)
                    labels.append(label_folder)
            label_map[label_folder] = label_index
            label_index += 1

    # 标签转换为索引
    labels = [label_map[label] for label in labels]

    print(f"类别标签映射：{labels__}")
    print(f"总样本数：{len(file_paths)}")

    # 两步划分法
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths,
        labels,
        test_size=(1 - train_ratio),
        random_state=random_state,
        stratify=labels,  # 保持类别比例
    )

    val_test_ratio = test_ratio / (test_ratio + val_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=val_test_ratio,
        random_state=random_state,
        stratify=temp_labels,
    )

    print(f"训练集：{len(train_paths)} 样本")
    print(f"验证集：{len(val_paths)} 样本")
    print(f"测试集：{len(test_paths)} 样本")

    # 创建数据集（FP16 版本）
    train_dataset = NPYDataset(train_paths, train_labels)
    validate_dataset = NPYDataset(val_paths, val_labels)
    test_dataset = NPYDataset(test_paths, test_labels)

    return train_dataset, validate_dataset, test_dataset, labels__
