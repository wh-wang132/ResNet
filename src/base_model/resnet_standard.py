#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标准 2D ResNet 架构（FP16 兼容版）
实现标准的 ResNet-18, ResNet-34, ResNet-50 架构
遵循标准 ResNet 规范，与现有代码框架兼容
"""

import torch
import torch.nn as nn
from torch.amp import autocast


class BasicBlock(nn.Module):
    """标准基础残差块（2D 版本，用于 ResNet-18/34）"""

    expansion = 1

    def __init__(
        self,
        in_channel,
        out_channel,
        stride=1,
        downsample=None,
        groups=1,
        width_per_group=64,
        dropout_p=0.0,
    ):
        super(BasicBlock, self).__init__()

        if groups != 1 or width_per_group != 64:
            raise ValueError("BasicBlock only supports groups=1 and width_per_group=64")

        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
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


class Bottleneck(nn.Module):
    """瓶颈残差块（2D 版本，用于 ResNet-50/101/152）"""

    expansion = 4

    def __init__(
        self,
        in_channel,
        out_channel,
        stride=1,
        downsample=None,
        groups=1,
        width_per_group=64,
        dropout_p=0.0,
    ):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.0)) * groups

        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=width,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(
            in_channels=width,
            out_channels=width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(
            in_channels=width,
            out_channels=out_channel * self.expansion,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
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
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet2D(nn.Module):
    """标准 2D ResNet 架构（ResNet-18/34/50）"""

    def __init__(
        self,
        block,
        blocks_num,
        num_classes=24,
        include_top=True,
        groups=1,
        width_per_group=64,
        dropout_p=0.0,
        in_channels=1,
        init_channels=64,
    ):
        """
        Args:
            block: 残差块类（BasicBlock 或 Bottleneck）
            blocks_num: 每层的残差块数量列表 [layer1, layer2, layer3, layer4]
            num_classes: 分类数（默认 24）
            include_top: 是否包含分类头
            groups: 分组卷积的组数（默认为 1）
            width_per_group: 每组宽度（默认 64）
            dropout_p: Dropout 概率（默认 0.0，与标准 ResNet 一致）
            in_channels: 输入通道数（默认 1，单通道灰度/特征）
            init_channels: 初始通道数（默认 64，与标准 ResNet 一致）
        """
        super(ResNet2D, self).__init__()
        self.include_top = include_top
        self.in_channel = init_channels
        self.groups = groups
        self.width_per_group = width_per_group

        # 初始卷积层（标准 ResNet 配置）
        self.conv1 = nn.Conv2d(
            in_channels,
            self.in_channel,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个标准残差层
        self.layer1 = self._make_layer(
            block, init_channels, blocks_num[0], dropout_p=dropout_p
        )
        self.layer2 = self._make_layer(
            block, init_channels * 2, blocks_num[1], stride=2, dropout_p=dropout_p
        )
        self.layer3 = self._make_layer(
            block, init_channels * 4, blocks_num[2], stride=2, dropout_p=dropout_p
        )
        self.layer4 = self._make_layer(
            block, init_channels * 8, blocks_num[3], stride=2, dropout_p=dropout_p
        )

        # 额外的 Dropout 层
        self.dropout = nn.Dropout(p=dropout_p)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(init_channels * 8 * block.expansion, num_classes)

        # Kaiming 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block, channel, block_num, stride=1, dropout_p=0.0):
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
                width_per_group=self.width_per_group,
                dropout_p=dropout_p,
            )
        )
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(
                block(
                    self.in_channel,
                    channel,
                    groups=self.groups,
                    width_per_group=self.width_per_group,
                    dropout_p=dropout_p,
                )
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
            x = self.layer4(x)

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

                x = self.layer4(x)
                if "layer4" in layer:
                    out["layer4"] = x

                return x if len(out) == 0 else out


def resnet18_2d(num_classes=24, dropout_p=0.0):
    """标准 ResNet-18 2D 版本"""
    return ResNet2D(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        init_channels=64,
        dropout_p=dropout_p,
    )


def resnet34_2d(num_classes=24, dropout_p=0.0):
    """标准 ResNet-34 2D 版本"""
    return ResNet2D(
        BasicBlock,
        [3, 4, 6, 3],
        num_classes=num_classes,
        init_channels=64,
        dropout_p=dropout_p,
    )


def resnet50_2d(num_classes=24, dropout_p=0.0):
    """标准 ResNet-50 2D 版本（使用 Bottleneck 块）"""
    return ResNet2D(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        init_channels=64,
        dropout_p=dropout_p,
    )
