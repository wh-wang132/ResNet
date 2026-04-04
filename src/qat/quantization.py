#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Torch 原生 FX QAT 配置与 prepare 逻辑。"""

import copy

import torch
from torch.ao.quantization import QConfig, QConfigMapping, disable_observer
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torch.ao.quantization.quantize_fx import prepare_qat_fx
from torch.nn.intrinsic.qat import freeze_bn_stats

from qat.utils import INPUT_SHAPE_NCHW


SUPPORTED_QAT_BACKEND = "torch_fx"
SUPPORTED_PREPARE_TYPE = "prepare_qat_fx"
QUANTIZATION_SCHEME_VERSION = 1


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


def create_qat_qconfig_mapping():
    activation_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        quant_min=0,
        quant_max=255,
    )
    weight_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        quant_min=-128,
        quant_max=127,
        ch_axis=0,
    )
    qconfig = QConfig(
        activation=activation_fake_quant,
        weight=weight_fake_quant,
    )
    return QConfigMapping().set_global(qconfig)


def build_quantization_meta(example_input_shape=None):
    normalized_shape = normalize_example_input_shape(example_input_shape)
    return {
        "quantization_scheme_version": QUANTIZATION_SCHEME_VERSION,
        "qat_backend": SUPPORTED_QAT_BACKEND,
        "prepare_type": SUPPORTED_PREPARE_TYPE,
        "example_input_shape": normalized_shape,
        "activation_observer": MovingAverageMinMaxObserver.__name__,
        "activation_dtype": str(torch.quint8),
        "activation_qscheme": str(torch.per_tensor_affine),
        "activation_quant_min": 0,
        "activation_quant_max": 255,
        "weight_observer": MovingAveragePerChannelMinMaxObserver.__name__,
        "weight_dtype": str(torch.qint8),
        "weight_qscheme": str(torch.per_channel_symmetric),
        "weight_quant_min": -128,
        "weight_quant_max": 127,
        "weight_ch_axis": 0,
        "weights": "per_channel_symmetric",
        "activations": "per_tensor",
        "convert_applied": False,
    }


def validate_quantization_meta(quantization_meta):
    if quantization_meta is None:
        raise ValueError("quantization_meta 不能为空")

    if quantization_meta.get("quantization_scheme_version") != QUANTIZATION_SCHEME_VERSION:
        raise ValueError("不支持的 quantization_scheme_version")
    if quantization_meta.get("qat_backend") != SUPPORTED_QAT_BACKEND:
        raise ValueError("当前仅支持 torch_fx QAT backend")
    if quantization_meta.get("prepare_type") != SUPPORTED_PREPARE_TYPE:
        raise ValueError("当前仅支持 prepare_qat_fx 恢复链")
    if quantization_meta.get("activation_observer") != MovingAverageMinMaxObserver.__name__:
        raise ValueError("activation_observer 与当前实现不一致")
    if quantization_meta.get("weight_observer") != MovingAveragePerChannelMinMaxObserver.__name__:
        raise ValueError("weight_observer 与当前实现不一致")
    if quantization_meta.get("activation_dtype") != str(torch.quint8):
        raise ValueError("activation_dtype 与当前实现不一致")
    if quantization_meta.get("weight_dtype") != str(torch.qint8):
        raise ValueError("weight_dtype 与当前实现不一致")
    if quantization_meta.get("activation_qscheme") != str(torch.per_tensor_affine):
        raise ValueError("activation_qscheme 与当前实现不一致")
    if quantization_meta.get("weight_qscheme") != str(torch.per_channel_symmetric):
        raise ValueError("weight_qscheme 与当前实现不一致")
    if int(quantization_meta.get("activation_quant_min", -1)) != 0:
        raise ValueError("activation_quant_min 与当前实现不一致")
    if int(quantization_meta.get("activation_quant_max", -1)) != 255:
        raise ValueError("activation_quant_max 与当前实现不一致")
    if int(quantization_meta.get("weight_quant_min", 1)) != -128:
        raise ValueError("weight_quant_min 与当前实现不一致")
    if int(quantization_meta.get("weight_quant_max", -1)) != 127:
        raise ValueError("weight_quant_max 与当前实现不一致")
    if int(quantization_meta.get("weight_ch_axis", -1)) != 0:
        raise ValueError("weight_ch_axis 与当前实现不一致")
    if bool(quantization_meta.get("convert_applied", False)):
        raise ValueError("当前 QAT checkpoint 不支持 convert_applied=True")
    return normalize_example_input_shape(quantization_meta.get("example_input_shape"))


def prepare_model_for_qat(model, device, quantization_meta=None, example_input_shape=None):
    model.train()
    if quantization_meta is not None:
        normalized_shape = validate_quantization_meta(quantization_meta)
    else:
        normalized_shape = normalize_example_input_shape(example_input_shape)
        quantization_meta = build_quantization_meta(normalized_shape)

    example_inputs = (build_example_inputs(device, normalized_shape),)
    qconfig_mapping = create_qat_qconfig_mapping()
    prepared_model = prepare_qat_fx(model, qconfig_mapping, example_inputs)
    return prepared_model, copy.deepcopy(quantization_meta), example_inputs


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
