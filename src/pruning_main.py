#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""剪枝 + 微调统一入口。"""

import copy
import json
import os

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

from base_model.dataset import data_set_split
from pruning.args import parse_args
from pruning.checkpoint import load_base_checkpoint
from pruning.evaluator import (
    count_model_stats,
    evaluate_model,
    evaluate_model_with_confusion_matrix,
)
from pruning.output import create_output_directory, save_summary
from pruning.pruner import prune_model
from pruning.topology import build_topology_metadata
from pruning.trainer import (
    append_round_best_info,
    finetune_model,
    save_pruned_checkpoint_without_finetune,
)
from pruning.utils import (
    INPUT_SHAPE_NCHW,
    build_compact_pruning_meta,
    create_optimized_dataloader,
    release_gpu_memory,
    setup_device,
)


def build_example_inputs(device):
    return torch.randn(*INPUT_SHAPE_NCHW, dtype=torch.float32, device=device)


def main():
    args = parse_args()
    print(args)

    release_gpu_memory()
    device = setup_device()

    model, checkpoint_meta, _ = load_base_checkpoint(
        args.model,
        device,
    )
    model_name = checkpoint_meta["model_name"]

    folder_path = create_output_directory(args, model_name)
    print(f"\n剪枝输出目录: {folder_path}")
    best_info_path = f"{folder_path}/best_pruned_info.txt"
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
        loader_name="剪枝训练集 DataLoader",
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
        loader_name="剪枝验证集 DataLoader",
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
        loader_name="剪枝测试集 DataLoader",
    )

    example_inputs = build_example_inputs(device)
    pre_prune_model = copy.deepcopy(model).to(device)
    baseline_stats = count_model_stats(pre_prune_model, example_inputs)
    baseline_val_metrics = evaluate_model(
        pre_prune_model,
        device,
        validate_loader,
        len(validate_dataset),
    )
    baseline_test_metrics = None
    if args.evaluate_test:
        baseline_test_metrics = evaluate_model(
            pre_prune_model,
            device,
            test_loader,
            len(test_dataset),
        )

    print(f"\n开始执行 iterative structured pruning，共 {args.pruning_steps} 轮...")
    rounds = []
    current_model = model
    final_pruning_meta = None
    final_topology_meta = None
    final_finetune_summary = None

    for step_index in range(1, args.pruning_steps + 1):
        print(f"\n开始第 {step_index}/{args.pruning_steps} 轮剪枝...")
        pruned_model, pruning_meta = prune_model(
            model=current_model,
            example_inputs=example_inputs,
            target_total_ratio=args.pruning_ratio,
            global_pruning=args.global_pruning,
            ignore_fc=args.ignore_fc,
            step_index=step_index,
            pruning_steps=args.pruning_steps,
        )
        compact_pruning_meta = build_compact_pruning_meta(pruning_meta, baseline_stats)

        topology_meta = build_topology_metadata(pruned_model)
        pruned_model.channel_cfg = topology_meta["channel_cfg"]

        pruned_val_metrics = evaluate_model(
            pruned_model,
            device,
            validate_loader,
            len(validate_dataset),
        )

        is_final_round = step_index == args.pruning_steps
        if args.finetune_epochs > 0:
            pruned_model, finetune_summary = finetune_model(
                model=pruned_model,
                device=device,
                train_loader=train_loader,
                validate_loader=validate_loader,
                val_num=len(validate_dataset),
                args=args,
                folder_path=folder_path,
                checkpoint_meta=checkpoint_meta,
                pruning_meta=compact_pruning_meta,
                initial_val_metrics=pruned_val_metrics,
                round_index=step_index,
                save_checkpoint=is_final_round,
            )
            append_round_best_info(best_info_path, step_index, finetune_summary)
        else:
            checkpoint_path = None
            if is_final_round:
                checkpoint_path = save_pruned_checkpoint_without_finetune(
                    model=pruned_model,
                    device=device,
                    folder_path=folder_path,
                    args=args,
                    checkpoint_meta=checkpoint_meta,
                    pruning_meta=compact_pruning_meta,
                    metrics=pruned_val_metrics,
                )
            finetune_summary = {
                "round_index": step_index,
                "best_acc": pruned_val_metrics["acc"],
                "best_val_loss": pruned_val_metrics["loss"],
                "best_epoch": 0,
                "checkpoint_path": checkpoint_path,
            }

        rounds.append(
            {
                "step_index": step_index,
                "step_ratio": pruning_meta["step_ratio"],
                "before_finetune": {
                    "val": pruned_val_metrics,
                    "topology": topology_meta,
                },
                "after_finetune": {
                    "val": {
                        "loss": finetune_summary["best_val_loss"],
                        "acc": finetune_summary["best_acc"],
                        "samples": len(validate_dataset),
                    },
                    "best_epoch": finetune_summary["best_epoch"],
                },
                "pruning_meta": compact_pruning_meta,
                "finetune_summary": finetune_summary,
            }
        )

        current_model = pruned_model
        final_pruning_meta = compact_pruning_meta
        final_topology_meta = topology_meta
        final_finetune_summary = finetune_summary

    final_test_metrics = None
    final_stats = count_model_stats(current_model, example_inputs)
    if args.evaluate_test:
        final_test_metrics = evaluate_model_with_confusion_matrix(
            current_model,
            device,
            test_loader,
            len(test_dataset),
            labels__,
            folder_path,
        )

    summary = {
        "model_name": model_name,
        "pruning_steps": args.pruning_steps,
        "labels": labels__,
        "baseline": {
            "val": baseline_val_metrics,
            "test": baseline_test_metrics,
            "stats": baseline_stats,
        },
        "rounds": rounds,
        "pruning_meta": final_pruning_meta,
        "finetune_summary": final_finetune_summary,
        "final": {
            "val": {
                "loss": final_finetune_summary["best_val_loss"],
                "acc": final_finetune_summary["best_acc"],
                "samples": len(validate_dataset),
            }
            if final_finetune_summary is not None
            else None,
            "test": final_test_metrics,
            "stats": final_stats,
        },
        "final_topology": final_topology_meta,
        "checkpoint_link_path": checkpoint_meta["checkpoint_link_path"],
        "resolved_checkpoint_path": checkpoint_meta["resolved_checkpoint_path"],
    }

    summary_path = save_summary(folder_path, summary)
    print("\n剪枝流程完成")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n摘要已保存至: {summary_path}")


if __name__ == "__main__":
    main()
