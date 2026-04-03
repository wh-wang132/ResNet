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


def build_example_inputs(device):
    return torch.randn(*INPUT_SHAPE_NCHW, dtype=torch.float32, device=device)


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


def build_quantization_meta():
    return {
        "quantization_scheme_version": 1,
        "qat_backend": "torch_fx",
        "weights": "per_channel_symmetric",
        "activations": "per_tensor",
        "convert_applied": False,
        "prepare_type": "prepare_qat_fx",
        "example_input_shape": list(INPUT_SHAPE_NCHW),
    }


def prepare_model_for_qat(model, device):
    model.train()
    example_inputs = (build_example_inputs(device),)
    qconfig_mapping = create_qat_qconfig_mapping()
    prepared_model = prepare_qat_fx(model, qconfig_mapping, example_inputs)
    return prepared_model, copy.deepcopy(build_quantization_meta()), example_inputs


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
