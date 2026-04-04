#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Torch 原生 FX QAT 统一入口。"""

import json
import os

import torch

from base_model.dataset import data_set_split
from base_model.plotting import configure_matplotlib
from qat.args import parse_args
from qat.checkpoint import load_pruning_checkpoint
from qat.evaluator import count_model_stats, evaluate_model, evaluate_model_with_confusion_matrix
from qat.output import create_output_directory, save_summary
from qat.quantization import build_example_inputs, prepare_model_for_qat
from qat.trainer import (
    finetune_qat_model,
    save_prepared_qat_checkpoint_without_finetune,
    write_best_qat_info,
)
from qat.utils import create_optimized_dataloader, release_gpu_memory, setup_device, to_repo_relative_path

configure_matplotlib()


def main():
    args = parse_args()
    print(args)

    release_gpu_memory()
    device = setup_device()

    float_model, checkpoint_meta, _ = load_pruning_checkpoint(args.pruning_checkpoint, device)
    model_name = checkpoint_meta["model_name"]

    folder_path = create_output_directory(args, checkpoint_meta)
    print(f"\nQAT 输出目录: {folder_path}")
    best_info_path = os.path.join(folder_path, "best_qat_info.txt")
    if os.path.exists(best_info_path):
        os.remove(best_info_path)

    train_dataset, validate_dataset, test_dataset, labels__ = data_set_split(
        args.data_dir,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        full_load=args.full_load,
        num_workers=args.num_workers,
        data_dtype=args.data_dtype,
    )

    pin_memory = args.pin_memory and torch.cuda.is_available()
    train_loader, _ = create_optimized_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=pin_memory,
        drop_last=True,
        loader_name="QAT 训练集 DataLoader",
    )
    validate_loader, _ = create_optimized_dataloader(
        validate_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=pin_memory,
        drop_last=False,
        loader_name="QAT 验证集 DataLoader",
    )
    test_loader, _ = create_optimized_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=pin_memory,
        drop_last=False,
        loader_name="QAT 测试集 DataLoader",
    )

    example_inputs = build_example_inputs(device)
    topology_meta = checkpoint_meta["quantization_source"]
    baseline_stats = count_model_stats(float_model, example_inputs)
    baseline_val_metrics = evaluate_model(
        float_model,
        device,
        validate_loader,
        len(validate_dataset),
    )
    baseline_test_metrics = None
    if args.evaluate_test:
        baseline_test_metrics = evaluate_model(
            float_model,
            device,
            test_loader,
            len(test_dataset),
        )

    prepared_model, quantization_meta, example_inputs = prepare_model_for_qat(float_model, device)
    prepared_model.channel_cfg = topology_meta["channel_cfg"]
    quantization_meta["example_input_shape"] = list(example_inputs[0].shape)

    if args.qat_epochs > 0:
        initial_qat_val_metrics = evaluate_model(
            prepared_model,
            device,
            validate_loader,
            len(validate_dataset),
        )
        prepared_model, finetune_summary = finetune_qat_model(
            model=prepared_model,
            device=device,
            train_loader=train_loader,
            validate_loader=validate_loader,
            val_num=len(validate_dataset),
            args=args,
            folder_path=folder_path,
            checkpoint_meta=checkpoint_meta,
            quantization_meta=quantization_meta,
            initial_val_metrics=initial_qat_val_metrics,
        )
    else:
        initial_qat_val_metrics = evaluate_model(
            prepared_model,
            device,
            validate_loader,
            len(validate_dataset),
        )
        checkpoint_path = save_prepared_qat_checkpoint_without_finetune(
            model=prepared_model,
            folder_path=folder_path,
            args=args,
            checkpoint_meta=checkpoint_meta,
            quantization_meta=quantization_meta,
            metrics=initial_qat_val_metrics,
        )
        finetune_summary = {
            "best_acc": initial_qat_val_metrics["acc"],
            "best_val_loss": initial_qat_val_metrics["loss"],
            "best_epoch": 0,
            "checkpoint_path": checkpoint_path,
        }
        write_best_qat_info(
            best_info_path=best_info_path,
            best_acc=finetune_summary["best_acc"],
            best_val_loss=finetune_summary["best_val_loss"],
            best_epoch=finetune_summary["best_epoch"],
        )

    final_test_metrics = None
    if args.evaluate_test:
        final_test_metrics = evaluate_model_with_confusion_matrix(
            prepared_model,
            device,
            test_loader,
            len(test_dataset),
            labels__,
            folder_path,
        )

    summary = {
        "model_name": model_name,
        "labels": labels__,
        "baseline": {
            "val": baseline_val_metrics,
            "test": baseline_test_metrics,
            "stats": baseline_stats,
        },
        "quantization_meta": quantization_meta,
        "finetune_summary": {
            **finetune_summary,
            "checkpoint_path": to_repo_relative_path(finetune_summary["checkpoint_path"]),
        },
        "final": {
            "val": {
                "loss": finetune_summary["best_val_loss"],
                "acc": finetune_summary["best_acc"],
                "samples": len(validate_dataset),
            },
            "test": final_test_metrics,
            "stats": baseline_stats,
        },
        "final_topology": topology_meta,
        "source_pruning_checkpoint_path": checkpoint_meta["source_pruning_checkpoint_path"],
    }

    summary_path = save_summary(folder_path, summary)
    print("\nQAT 流程完成")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n摘要已保存至: {summary_path}")


if __name__ == "__main__":
    main()
