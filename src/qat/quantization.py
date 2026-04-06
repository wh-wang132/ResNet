#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Torch 原生 FX QAT 配置与 prepare 逻辑。"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import QConfig, QConfigMapping, disable_observer
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torch.ao.quantization.quantize_fx import prepare_qat_fx
from torch.nn.intrinsic.qat import freeze_bn_stats

from qat.utils import INPUT_SHAPE_NCHW


QUANTIZATION_SCHEME_VERSION = 3
CANONICAL_QAT_SCHEME_NAME = "torch_fx_qat_cann_v1"

QCONFIG_OBJECT_TYPE_TARGETS = (
    (nn.Conv2d, "conv"),
    (nn.BatchNorm2d, "conv"),
    (nn.ReLU, "conv"),
    (F.relu, "conv"),
    (nn.Linear, "linear"),
    (F.linear, "linear"),
)


def normalize_example_input_shape(example_input_shape):
    if example_input_shape is None:
        example_input_shape = INPUT_SHAPE_NCHW

    if isinstance(example_input_shape, torch.Size):
        example_input_shape = list(example_input_shape)

    if not isinstance(example_input_shape, (list, tuple)):
        raise TypeError("example_input_shape 必须是 list/tuple/torch.Size")

    normalized_shape = [int(dim) for dim in example_input_shape]
    if len(normalized_shape) != 4:
        raise ValueError("example_input_shape 必须是 NCHW 四维形状")
    if any(dim <= 0 for dim in normalized_shape):
        raise ValueError("example_input_shape 的每一维都必须大于 0")
    return normalized_shape


def build_example_inputs(device, example_input_shape=None):
    normalized_shape = normalize_example_input_shape(example_input_shape)
    return torch.randn(*normalized_shape, dtype=torch.float32, device=device)


def _build_qconfig(activation_fake_quant, weight_fake_quant):
    return QConfig(
        activation=activation_fake_quant,
        weight=weight_fake_quant,
    )


def _build_activation_fake_quant():
    return FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        quant_min=0,
        quant_max=255,
    )


def _build_conv_weight_fake_quant():
    return FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        quant_min=-128,
        quant_max=127,
        ch_axis=0,
    )


def _build_linear_weight_fake_quant():
    return FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        quant_min=-128,
        quant_max=127,
    )


def create_canonical_qat_qconfig_mapping():
    activation_fake_quant = _build_activation_fake_quant()
    qconfigs = {
        "conv": _build_qconfig(activation_fake_quant, _build_conv_weight_fake_quant()),
        "linear": _build_qconfig(activation_fake_quant, _build_linear_weight_fake_quant()),
    }

    qconfig_mapping = QConfigMapping().set_global(qconfigs["conv"])
    for object_type, qconfig_name in QCONFIG_OBJECT_TYPE_TARGETS:
        qconfig_mapping = qconfig_mapping.set_object_type(object_type, qconfigs[qconfig_name])
    return qconfig_mapping


def build_quantization_meta(example_input_shape=None):
    normalized_shape = normalize_example_input_shape(example_input_shape)
    return {
        "quantization_scheme_version": QUANTIZATION_SCHEME_VERSION,
        "scheme_name": CANONICAL_QAT_SCHEME_NAME,
        "example_input_shape": normalized_shape,
    }


def validate_quantization_meta(quantization_meta):
    if quantization_meta is None:
        raise ValueError("quantization_meta 不能为空")

    if int(quantization_meta.get("quantization_scheme_version", -1)) != QUANTIZATION_SCHEME_VERSION:
        raise ValueError(
            "当前 QAT checkpoint 使用旧版 quantization_meta 契约，请重新跑一遍新的 QAT"
        )
    if quantization_meta.get("scheme_name") != CANONICAL_QAT_SCHEME_NAME:
        raise ValueError("当前 QAT checkpoint 的 scheme_name 与当前实现不兼容，请重新跑一遍新的 QAT")
    return normalize_example_input_shape(quantization_meta.get("example_input_shape"))


def prepare_model_for_qat(model, device, quantization_meta=None, example_input_shape=None):
    model.train()
    if quantization_meta is not None:
        normalized_shape = validate_quantization_meta(quantization_meta)
        canonical_meta = build_quantization_meta(normalized_shape)
    else:
        normalized_shape = normalize_example_input_shape(example_input_shape)
        canonical_meta = build_quantization_meta(normalized_shape)

    qconfig_mapping = create_canonical_qat_qconfig_mapping()
    example_inputs = (build_example_inputs(device, normalized_shape),)
    prepared_model = prepare_qat_fx(model, qconfig_mapping, example_inputs)
    return prepared_model, copy.deepcopy(canonical_meta), example_inputs


def maybe_apply_qat_freeze_policy(model, epoch_index, total_epochs, freeze_state):
    current_epoch = epoch_index + 1
    if total_epochs > 0 and not freeze_state["bn_frozen"] and current_epoch > (total_epochs / 2):
        model.apply(freeze_bn_stats)
        freeze_state["bn_frozen"] = True

    observer_disable_threshold = max(int(total_epochs * 0.8), 1)
    if total_epochs > 0 and not freeze_state["observer_frozen"] and current_epoch > observer_disable_threshold:
        model.apply(disable_observer)
        freeze_state["observer_frozen"] = True

    return freeze_state
