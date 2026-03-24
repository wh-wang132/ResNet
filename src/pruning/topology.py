#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""剪枝后模型拓扑导出。"""

from pruning.utils import build_architecture_signature, get_raw_model


def _extract_downsample_cfg(downsample):
    if downsample is None:
        return None
    conv = downsample[0]
    return {
        "out_channels": int(conv.out_channels),
        "stride": int(conv.stride[0]),
    }


def _extract_standard_channel_cfg(model):
    cfg = {
        "block_type": model.layer1[0].__class__.__name__,
        "stem": {
            "in_channels": int(model.conv1.in_channels),
            "out_channels": int(model.conv1.out_channels),
        },
        "layers": [],
        "fc": None,
        "include_top": bool(model.include_top),
        "groups": int(getattr(model, "groups", 1)),
        "width_per_group": int(getattr(model, "width_per_group", 64)),
    }

    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        layer_cfg = {"blocks": []}
        for block in layer:
            if hasattr(block, "conv3"):
                out_channels = int(block.conv3.out_channels // block.expansion)
                mid_channels = int(block.conv1.out_channels)
                stride = int(block.conv2.stride[0])
            else:
                out_channels = int(block.conv2.out_channels)
                mid_channels = int(block.conv1.out_channels)
                stride = int(block.conv1.stride[0])

            layer_cfg["blocks"].append(
                {
                    "in_channels": int(block.conv1.in_channels),
                    "mid_channels": mid_channels,
                    "out_channels": out_channels,
                    "stride": stride,
                    "downsample": _extract_downsample_cfg(block.downsample),
                }
            )
        cfg["layers"].append(layer_cfg)

    if model.include_top:
        cfg["fc"] = {
            "in_features": int(model.fc.in_features),
            "out_features": int(model.fc.out_features),
        }

    return cfg


def _extract_lightweight_channel_cfg(model):
    cfg = {
        "block_type": model.layer1[0].__class__.__name__,
        "stem": {
            "in_channels": int(model.conv1.in_channels),
            "out_channels": int(model.conv1.out_channels),
        },
        "layers": [],
        "fc": None,
        "include_top": bool(model.include_top),
        "groups": int(getattr(model, "groups", 1)),
        "width_per_group": int(getattr(model, "width_per_group", 16)),
    }

    for layer_name in ["layer1", "layer2", "layer3"]:
        layer = getattr(model, layer_name)
        layer_cfg = {"blocks": []}
        for block in layer:
            layer_cfg["blocks"].append(
                {
                    "in_channels": int(block.conv1.in_channels),
                    "mid_channels": int(block.conv1.out_channels),
                    "out_channels": int(block.conv2.out_channels),
                    "stride": int(block.conv1.stride[0]),
                    "downsample": _extract_downsample_cfg(block.downsample),
                }
            )
        cfg["layers"].append(layer_cfg)

    if model.include_top:
        cfg["fc"] = {
            "in_features": int(model.fc.in_features),
            "out_features": int(model.fc.out_features),
        }

    return cfg


def extract_model_channel_cfg(model):
    raw_model = get_raw_model(model)
    if hasattr(raw_model, "layer4"):
        return _extract_standard_channel_cfg(raw_model)
    if hasattr(raw_model, "layer3"):
        return _extract_lightweight_channel_cfg(raw_model)
    raise ValueError(f"暂不支持从该模型导出 channel_cfg: {raw_model.__class__.__name__}")


def build_topology_metadata(model):
    raw_model = get_raw_model(model)
    channel_cfg = extract_model_channel_cfg(raw_model)
    architecture_signature = build_architecture_signature(raw_model)
    raw_model.channel_cfg = channel_cfg
    return {
        "channel_cfg": channel_cfg,
        "architecture_signature": architecture_signature,
    }
