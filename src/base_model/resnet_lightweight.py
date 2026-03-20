#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级 2D ResNet 架构（FP16 兼容版）
专为.npy 数据集设计
显著减少内存消耗和过拟合
支持 FP16 训练与推理
"""

import torch
import torch.nn as nn
from torch.amp import autocast


class LightweightBasicBlock2D(nn.Module):
    """轻量级基础残差块（2D 版本）"""

    expansion = 1

    def __init__(
        self,
        in_channel,
        out_channel,
        stride=1,
        downsample=None,
        groups=1,
        dropout_p=0.2,
    ):
        super(LightweightBasicBlock2D, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else None
        self.downsample = downsample

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out += identity
        out = self.relu(out)

        return out


class LightweightResNet2D(nn.Module):
    """轻量级 2D ResNet 架构"""

    def __init__(
        self,
        block,
        blocks_num,
        num_classes=24,
        include_top=True,
        groups=1,
        width_per_group=16,
        dropout_p=0.2,
        in_channels=1,
        init_channels=32,
    ):
        """
        Args:
            block: 残差块类
            blocks_num: 每层的残差块数量列表
            num_classes: 分类数（默认 24）
            include_top: 是否包含分类头
            groups: 分组卷积的组数（默认为 1（无分组）
            width_per_group: 每组宽度
            dropout_p: Dropout 概率（缓解过拟合）
            in_channels: 输入通道数（默认 1，单通道灰度/特征）
            init_channels: 初始通道数（比 ResNet-18 的 64 减小为 32）
        """
        super(LightweightResNet2D, self).__init__()
        self.include_top = include_top
        self.in_channel = init_channels
        self.groups = groups
        self.width_per_group = width_per_group

        # 初始卷积层（使用更小的通道数和核）
        self.conv1 = nn.Conv2d(
            in_channels,
            self.in_channel,
            kernel_size=5,  # 比 7 更小
            stride=2,
            padding=2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个残差层（使用更少的块）
        self.layer1 = self._make_layer(
            block, init_channels, blocks_num[0], dropout_p=dropout_p
        )
        self.layer2 = self._make_layer(
            block, init_channels * 2, blocks_num[1], stride=2, dropout_p=dropout_p
        )
        self.layer3 = self._make_layer(
            block, init_channels * 4, blocks_num[2], stride=2, dropout_p=dropout_p
        )

        # 移除 layer4 以减少复杂度
        # 额外的 Dropout 层
        self.dropout = nn.Dropout(p=dropout_p)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(init_channels * 4 * block.expansion, num_classes)

        # Kaiming 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block, channel, block_num, stride=1, dropout_p=0.2):
        """构建残差层"""
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channel,
                    channel * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    groups=self.groups,
                ),
                nn.BatchNorm2d(channel * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.in_channel,
                channel,
                downsample=downsample,
                stride=stride,
                groups=self.groups,
                dropout_p=dropout_p,
            )
        )
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(
                block(self.in_channel, channel, groups=self.groups, dropout_p=dropout_p)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        with autocast("cuda", enabled=torch.cuda.is_available()):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            if self.include_top:
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.dropout(x)
                x = self.fc(x)

            return x

    def get_features(self, x, layer=None):
        """提取中间层特征（FP16 兼容）"""
        with autocast("cuda", enabled=torch.cuda.is_available()):
            if layer is None:
                return self.forward(x)
            else:
                out = {}
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                if "layer1" in layer:
                    out["layer1"] = x

                x = self.layer2(x)
                if "layer2" in layer:
                    out["layer2"] = x

                x = self.layer3(x)
                if "layer3" in layer:
                    out["layer3"] = x

                return x if len(out) == 0 else out


def resnet6_2d(num_classes=24, dropout_p=0.2):
    """超轻量 ResNet-6 2D 版本（只有 2 个残差层）"""
    return LightweightResNet2D(
        LightweightBasicBlock2D,
        [1, 1, 1],
        num_classes=num_classes,
        init_channels=32,
        dropout_p=dropout_p,
    )


def resnet10_2d(num_classes=24, dropout_p=0.2):
    """轻量 ResNet-10 2D 版本"""
    return LightweightResNet2D(
        LightweightBasicBlock2D,
        [1, 1, 1],
        num_classes=num_classes,
        init_channels=48,
        dropout_p=dropout_p,
    )


def resnet14_2d(num_classes=24, dropout_p=0.2):
    """中等 ResNet-14 2D 版本"""
    return LightweightResNet2D(
        LightweightBasicBlock2D,
        [2, 2, 1],
        num_classes=num_classes,
        init_channels=48,
        dropout_p=dropout_p,
    )
