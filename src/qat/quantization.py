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


SUPPORTED_QAT_BACKEND = "torch_fx"
SUPPORTED_PREPARE_TYPE = "prepare_qat_fx"
QUANTIZATION_SCHEME_VERSION = 2

DTYPE_NAME_MAP = {
    str(torch.quint8): torch.quint8,
    str(torch.qint8): torch.qint8,
}

QSCHEME_NAME_MAP = {
    str(torch.per_tensor_affine): torch.per_tensor_affine,
    str(torch.per_tensor_symmetric): torch.per_tensor_symmetric,
    str(torch.per_channel_symmetric): torch.per_channel_symmetric,
}

OBSERVER_NAME_MAP = {
    MovingAverageMinMaxObserver.__name__: MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver.__name__: MovingAveragePerChannelMinMaxObserver,
}


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


def _build_fake_quant(observer_name, dtype_name, qscheme_name, quant_min, quant_max, ch_axis=None):
    observer = OBSERVER_NAME_MAP[observer_name]
    dtype = DTYPE_NAME_MAP[dtype_name]
    qscheme = QSCHEME_NAME_MAP[qscheme_name]
    fake_quant_args = {
        "observer": observer,
        "dtype": dtype,
        "qscheme": qscheme,
        "quant_min": int(quant_min),
        "quant_max": int(quant_max),
    }
    if ch_axis is not None:
        fake_quant_args["ch_axis"] = int(ch_axis)
    return FusedMovingAvgObsFakeQuantize.with_args(**fake_quant_args)


def _build_activation_fake_quant(quantization_meta):
    return _build_fake_quant(
        observer_name=quantization_meta["activation_observer"],
        dtype_name=quantization_meta["activation_dtype"],
        qscheme_name=quantization_meta["activation_qscheme"],
        quant_min=quantization_meta["activation_quant_min"],
        quant_max=quantization_meta["activation_quant_max"],
    )


def _build_weight_fake_quant(quantization_meta, prefix):
    ch_axis = quantization_meta.get(f"{prefix}_ch_axis")
    return _build_fake_quant(
        observer_name=quantization_meta[f"{prefix}_observer"],
        dtype_name=quantization_meta[f"{prefix}_dtype"],
        qscheme_name=quantization_meta[f"{prefix}_qscheme"],
        quant_min=quantization_meta[f"{prefix}_quant_min"],
        quant_max=quantization_meta[f"{prefix}_quant_max"],
        ch_axis=ch_axis,
    )


def create_qat_qconfig_mapping_from_meta(quantization_meta):
    activation_fake_quant = _build_activation_fake_quant(quantization_meta)
    conv_weight_fake_quant = _build_weight_fake_quant(quantization_meta, "conv_weight")
    linear_weight_fake_quant = _build_weight_fake_quant(quantization_meta, "linear_weight")

    conv_qconfig = QConfig(
        activation=activation_fake_quant,
        weight=conv_weight_fake_quant,
    )
    linear_qconfig = QConfig(
        activation=activation_fake_quant,
        weight=linear_weight_fake_quant,
    )
    return (
        QConfigMapping()
        .set_global(conv_qconfig)
        .set_object_type(nn.Conv2d, conv_qconfig)
        .set_object_type(nn.BatchNorm2d, conv_qconfig)
        .set_object_type(nn.ReLU, conv_qconfig)
        .set_object_type(F.relu, conv_qconfig)
        .set_object_type(nn.Linear, linear_qconfig)
        .set_object_type(F.linear, linear_qconfig)
    )


def create_default_qat_qconfig_mapping():
    return create_qat_qconfig_mapping_from_meta(build_quantization_meta())


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
        "conv_weight_observer": MovingAveragePerChannelMinMaxObserver.__name__,
        "conv_weight_dtype": str(torch.qint8),
        "conv_weight_qscheme": str(torch.per_channel_symmetric),
        "conv_weight_quant_min": -128,
        "conv_weight_quant_max": 127,
        "conv_weight_ch_axis": 0,
        "linear_weight_observer": MovingAverageMinMaxObserver.__name__,
        "linear_weight_dtype": str(torch.qint8),
        "linear_weight_qscheme": str(torch.per_tensor_symmetric),
        "linear_weight_quant_min": -128,
        "linear_weight_quant_max": 127,
        "weights": "conv_per_channel_linear_per_tensor",
        "activations": "per_tensor",
        "convert_applied": False,
    }


def _validate_weight_quantization_meta(quantization_meta, prefix, expected):
    observer_key = f"{prefix}_observer"
    dtype_key = f"{prefix}_dtype"
    qscheme_key = f"{prefix}_qscheme"
    quant_min_key = f"{prefix}_quant_min"
    quant_max_key = f"{prefix}_quant_max"
    ch_axis_key = f"{prefix}_ch_axis"

    if quantization_meta.get(observer_key) not in OBSERVER_NAME_MAP:
        raise ValueError(f"不支持的 {observer_key}")
    if quantization_meta.get(dtype_key) not in DTYPE_NAME_MAP:
        raise ValueError(f"不支持的 {dtype_key}")
    if quantization_meta.get(qscheme_key) not in QSCHEME_NAME_MAP:
        raise ValueError(f"不支持的 {qscheme_key}")
    if quantization_meta.get(observer_key) != expected["observer"]:
        raise ValueError(f"{observer_key} 与当前实现不一致")
    if quantization_meta.get(dtype_key) != expected["dtype"]:
        raise ValueError(f"{dtype_key} 与当前实现不一致")
    if quantization_meta.get(qscheme_key) != expected["qscheme"]:
        raise ValueError(f"{qscheme_key} 与当前实现不一致")
    if int(quantization_meta.get(quant_min_key, 1)) != expected["quant_min"]:
        raise ValueError(f"{quant_min_key} 与当前实现不一致")
    if int(quantization_meta.get(quant_max_key, -1)) != expected["quant_max"]:
        raise ValueError(f"{quant_max_key} 与当前实现不一致")

    expected_ch_axis = expected.get("ch_axis")
    if expected_ch_axis is None:
        if ch_axis_key in quantization_meta:
            raise ValueError(f"{ch_axis_key} 在当前实现中不应存在")
    else:
        if int(quantization_meta.get(ch_axis_key, -1)) != expected_ch_axis:
            raise ValueError(f"{ch_axis_key} 与当前实现不一致")


def validate_quantization_meta(quantization_meta):
    if quantization_meta is None:
        raise ValueError("quantization_meta 不能为空")

    if quantization_meta.get("quantization_scheme_version") != QUANTIZATION_SCHEME_VERSION:
        raise ValueError(
            "当前 CANN 兼容 QAT 导出仅支持 quantization_scheme_version=2，请重新跑一遍新的 QAT"
        )
    if quantization_meta.get("qat_backend") != SUPPORTED_QAT_BACKEND:
        raise ValueError("当前仅支持 torch_fx QAT backend")
    if quantization_meta.get("prepare_type") != SUPPORTED_PREPARE_TYPE:
        raise ValueError("当前仅支持 prepare_qat_fx 恢复链")
    if quantization_meta.get("activation_observer") not in OBSERVER_NAME_MAP:
        raise ValueError("不支持的 activation_observer")
    if quantization_meta.get("activation_dtype") not in DTYPE_NAME_MAP:
        raise ValueError("不支持的 activation_dtype")
    if quantization_meta.get("activation_qscheme") not in QSCHEME_NAME_MAP:
        raise ValueError("不支持的 activation_qscheme")
    if quantization_meta.get("activation_observer") != MovingAverageMinMaxObserver.__name__:
        raise ValueError("activation_observer 与当前实现不一致")
    if quantization_meta.get("activation_dtype") != str(torch.quint8):
        raise ValueError("activation_dtype 与当前实现不一致")
    if quantization_meta.get("activation_qscheme") != str(torch.per_tensor_affine):
        raise ValueError("activation_qscheme 与当前实现不一致")
    if int(quantization_meta.get("activation_quant_min", -1)) != 0:
        raise ValueError("activation_quant_min 与当前实现不一致")
    if int(quantization_meta.get("activation_quant_max", -1)) != 255:
        raise ValueError("activation_quant_max 与当前实现不一致")

    _validate_weight_quantization_meta(
        quantization_meta,
        "conv_weight",
        {
            "observer": MovingAveragePerChannelMinMaxObserver.__name__,
            "dtype": str(torch.qint8),
            "qscheme": str(torch.per_channel_symmetric),
            "quant_min": -128,
            "quant_max": 127,
            "ch_axis": 0,
        },
    )
    _validate_weight_quantization_meta(
        quantization_meta,
        "linear_weight",
        {
            "observer": MovingAverageMinMaxObserver.__name__,
            "dtype": str(torch.qint8),
            "qscheme": str(torch.per_tensor_symmetric),
            "quant_min": -128,
            "quant_max": 127,
        },
    )

    if bool(quantization_meta.get("convert_applied", False)):
        raise ValueError("当前 QAT checkpoint 不支持 convert_applied=True")
    return normalize_example_input_shape(quantization_meta.get("example_input_shape"))


def prepare_model_for_qat(model, device, quantization_meta=None, example_input_shape=None):
    model.train()
    if quantization_meta is not None:
        normalized_shape = validate_quantization_meta(quantization_meta)
        qconfig_mapping = create_qat_qconfig_mapping_from_meta(quantization_meta)
    else:
        normalized_shape = normalize_example_input_shape(example_input_shape)
        quantization_meta = build_quantization_meta(normalized_shape)
        qconfig_mapping = create_qat_qconfig_mapping_from_meta(quantization_meta)

    example_inputs = (build_example_inputs(device, normalized_shape),)
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
