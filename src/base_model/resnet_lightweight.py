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
import copy


def build_lightweight_resnet2d_channel_cfg(
    block,
    blocks_num,
    init_channels=32,
    in_channels=1,
    groups=1,
    include_top=True,
    num_classes=24,
):
    """构建轻量 ResNet 的默认逐层通道配置。"""
    cfg = {
        "block_type": block.__name__,
        "stem": {"in_channels": in_channels, "out_channels": init_channels},
        "layers": [],
        "fc": None,
        "include_top": include_top,
        "groups": groups,
    }

    in_ch = init_channels
    stage_channels = [init_channels, init_channels * 2, init_channels * 4]

    for stage_idx, (block_num, out_channel) in enumerate(zip(blocks_num, stage_channels)):
        layer_cfg = {"blocks": []}
        for block_idx in range(block_num):
            stride = 2 if stage_idx > 0 and block_idx == 0 else 1
            downsample = None
            if stride != 1 or in_ch != out_channel * block.expansion:
                downsample = {
                    "out_channels": out_channel * block.expansion,
                    "stride": stride,
                }

            block_cfg = {
                "in_channels": in_ch,
                "mid_channels": out_channel,
                "out_channels": out_channel,
                "stride": stride,
                "downsample": downsample,
            }
            layer_cfg["blocks"].append(block_cfg)
            in_ch = out_channel * block.expansion
        cfg["layers"].append(layer_cfg)

    if include_top:
        cfg["fc"] = {
            "in_features": in_ch,
            "out_features": num_classes,
        }

    return cfg


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
        mid_channels=None,
        downsample_out_channels=None,
    ):
        super().__init__()

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
        channel_cfg=None,
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
        super().__init__()
        self.include_top = include_top
        self.groups = groups
        self.width_per_group = width_per_group

        if channel_cfg is None:
            channel_cfg = build_lightweight_resnet2d_channel_cfg(
                block=block,
                blocks_num=blocks_num,
                init_channels=init_channels,
                in_channels=in_channels,
                groups=groups,
                include_top=include_top,
                num_classes=num_classes,
            )
        self.channel_cfg = copy.deepcopy(channel_cfg)
        self.include_top = self.channel_cfg.get("include_top", include_top)

        stem_cfg = self.channel_cfg["stem"]

        # 初始卷积层（使用更小的通道数和核）
        self.conv1 = nn.Conv2d(
            stem_cfg["in_channels"],
            stem_cfg["out_channels"],
            kernel_size=5,  # 比 7 更小
            stride=2,
            padding=2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(stem_cfg["out_channels"])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个残差层（使用更少的块）
        self.layer1 = self._make_layer_from_cfg(
            block, self.channel_cfg["layers"][0], dropout_p=dropout_p
        )
        self.layer2 = self._make_layer_from_cfg(
            block, self.channel_cfg["layers"][1], dropout_p=dropout_p
        )
        self.layer3 = self._make_layer_from_cfg(
            block, self.channel_cfg["layers"][2], dropout_p=dropout_p
        )

        # 移除 layer4 以减少复杂度
        # 额外的 Dropout 层
        self.dropout = nn.Dropout(p=dropout_p)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            fc_cfg = self.channel_cfg["fc"]
            self.fc = nn.Linear(fc_cfg["in_features"], fc_cfg["out_features"])

        # Kaiming 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_downsample(self, in_channels, downsample_cfg):
        if downsample_cfg is None:
            return None
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                downsample_cfg["out_channels"],
                kernel_size=1,
                stride=downsample_cfg["stride"],
                bias=False,
                groups=self.groups,
            ),
            nn.BatchNorm2d(downsample_cfg["out_channels"]),
        )

    def _make_layer_from_cfg(self, block, layer_cfg, dropout_p=0.2):
        """按逐层通道配置构建残差层。"""
        layers = []
        for block_cfg in layer_cfg["blocks"]:
            downsample = self._make_downsample(
                block_cfg["in_channels"], block_cfg.get("downsample")
            )
            layers.append(
                block(
                    block_cfg["in_channels"],
                    block_cfg["out_channels"],
                    downsample=downsample,
                    stride=block_cfg["stride"],
                    groups=self.groups,
                    dropout_p=dropout_p,
                    mid_channels=block_cfg.get("mid_channels"),
                    downsample_out_channels=(
                        None
                        if block_cfg.get("downsample") is None
                        else block_cfg["downsample"]["out_channels"]
                    ),
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
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


def lightweight_resnet2d_from_cfg(
    blocks_num,
    channel_cfg,
    num_classes=24,
    dropout_p=0.2,
    include_top=True,
    in_channels=1,
):
    return LightweightResNet2D(
        LightweightBasicBlock2D,
        blocks_num,
        num_classes=num_classes,
        include_top=channel_cfg.get("include_top", include_top),
        in_channels=in_channels,
        init_channels=channel_cfg["stem"]["out_channels"],
        dropout_p=dropout_p,
        channel_cfg=channel_cfg,
    )


def resnet6_2d_from_cfg(
    channel_cfg, num_classes=24, dropout_p=0.2, include_top=True, in_channels=1
):
    return lightweight_resnet2d_from_cfg(
        [1, 1, 1],
        channel_cfg,
        num_classes=num_classes,
        dropout_p=dropout_p,
        include_top=include_top,
        in_channels=in_channels,
    )


def resnet10_2d_from_cfg(
    channel_cfg, num_classes=24, dropout_p=0.2, include_top=True, in_channels=1
):
    return lightweight_resnet2d_from_cfg(
        [1, 1, 1],
        channel_cfg,
        num_classes=num_classes,
        dropout_p=dropout_p,
        include_top=include_top,
        in_channels=in_channels,
    )


def resnet14_2d_from_cfg(
    channel_cfg, num_classes=24, dropout_p=0.2, include_top=True, in_channels=1
):
    return lightweight_resnet2d_from_cfg(
        [2, 2, 1],
        channel_cfg,
        num_classes=num_classes,
        dropout_p=dropout_p,
        include_top=include_top,
        in_channels=in_channels,
    )
