#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文插图绘制函数。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math

from base_model.plotting import configure_matplotlib

from .contracts import FigureRecord, natural_key
from .output import write_csv

configure_matplotlib()
import matplotlib.pyplot as plt


MODEL_COLORS = {
    "resnet6_2d": "#0072B2",
    "resnet10_2d": "#D55E00",
    "resnet14_2d": "#009E73",
    "resnet18_2d": "#CC79A7",
    "resnet34_2d": "#E69F00",
}


def write_all_outputs(records, output_dir, formats):
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    records = list(records)

    figures: list[Path] = []
    tables: list[Path] = []
    tables.append(write_csv(tables_dir / "records.csv", [record.to_row() for record in records]))
    tables.append(write_csv(tables_dir / "pruning_tradeoff.csv", _pruning_rows(records)))
    tables.append(write_csv(tables_dir / "stage_accuracy_summary.csv", _stage_error_rows(records)))
    tables.append(write_csv(tables_dir / "onnx_metric_delta.csv", _onnx_delta_rows(records)))
    tables.append(write_csv(tables_dir / "atc_amct_interface_matrix.csv", _interface_rows(records)))

    figures.extend(plot_pruning_accuracy_complexity(records, output_dir, formats))
    figures.extend(plot_compression_by_model(records, output_dir, formats))
    figures.extend(plot_stage_accuracy_flow(records, output_dir, formats))
    figures.extend(plot_onnx_metric_delta(records, output_dir, formats))
    figures.extend(plot_interface_matrix(records, output_dir, formats))
    return figures, tables


def plot_pruning_accuracy_complexity(records, output_dir, formats):
    rows = _pruning_rows(records)
    if not rows:
        return []

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    plotted_error = False
    plotted_params = False
    for model_name, model_rows in _group_rows(rows, "model_name").items():
        model_rows = sorted(model_rows, key=lambda row: row["pruning_ratio"])
        color = _color_for_model(model_name)
        error_points = [
            (row["pruning_ratio"] * 100, row["error_rate"] * 100)
            for row in model_rows
            if _is_positive(row.get("error_rate"))
        ]
        params_points = [
            (row["pruning_ratio"] * 100, row["params_remaining_ratio"] * 100)
            for row in model_rows
            if _is_positive(row.get("params_remaining_ratio"))
        ]
        if error_points:
            x_values, error_values = zip(*error_points)
            axes[0].plot(
                x_values,
                error_values,
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=model_name,
                color=color,
            )
            plotted_error = True
        if params_points:
            x_values, params_values = zip(*params_points)
            axes[1].plot(
                x_values,
                params_values,
                marker="s",
                linewidth=1.8,
                markersize=4,
                label=model_name,
                color=color,
            )
            plotted_params = True

    if not plotted_error and not plotted_params:
        plt.close(fig)
        return []

    axes[0].set_yscale("log")
    axes[0].set_ylabel("Error Rate (%)")
    axes[0].set_title("Pruning Error Rate and Parameter Trade-off")
    axes[0].grid(True, alpha=0.3, linestyle="--")
    if plotted_error:
        axes[0].legend(ncol=2)
    axes[1].set_xlabel("Target Pruning Ratio (%)")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Parameters Remaining (%)")
    axes[1].grid(True, alpha=0.3, linestyle="--")
    if plotted_params:
        axes[1].legend(ncol=2)
    _set_ylim_with_padding(axes[0])
    _set_ylim_with_padding(axes[1])
    return _save_figure(fig, output_dir, "fig1_pruning_accuracy_complexity", formats)


def plot_compression_by_model(records, output_dir, formats):
    pruning_records = [record for record in records if record.stage == "pruning"]
    best_records = []
    for _, model_records in _group_records(pruning_records, "model_name").items():
        candidates = [
            record
            for record in model_records
            if _is_positive(record.params)
            and _is_positive(record.baseline_params)
            and _is_positive(record.macs)
            and _is_positive(record.baseline_macs)
        ]
        if candidates:
            best_records.append(min(candidates, key=lambda item: item.params or math.inf))
    if not best_records:
        return []

    best_records.sort(key=lambda item: natural_key(item.model_name))
    labels = [record.model_name.replace("_2d", "") for record in best_records]
    x_values = list(range(len(best_records)))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    baseline_params = [(record.baseline_params or 0) / 1e6 for record in best_records]
    final_params = [(record.params or 0) / 1e6 for record in best_records]
    baseline_macs = [(record.baseline_macs or 0) / 1e9 for record in best_records]
    final_macs = [(record.macs or 0) / 1e9 for record in best_records]

    _paired_bars(
        axes[0],
        x_values,
        baseline_params,
        final_params,
        labels,
        width,
        "Parameters (M)",
        "Best Compression by Model",
        yscale="log",
    )
    _paired_bars(
        axes[1],
        x_values,
        baseline_macs,
        final_macs,
        labels,
        width,
        "MACs (G)",
        "Computation Reduction by Model",
        yscale="log",
    )
    return _save_figure(fig, output_dir, "fig2_compression_by_model", formats)


def plot_stage_accuracy_flow(records, output_dir, formats):
    rows = [
        row
        for row in _stage_error_rows(records)
        if _is_positive(row.get("mean_error_rate"))
    ]
    if not rows:
        return []

    labels = [row["stage_label"] for row in rows]
    values = [row["mean_error_rate"] * 100 for row in rows]
    counts = [row["count"] for row in rows]
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(labels, values, color=colors[: len(values)], width=0.62)
    ax.set_yscale("log")
    ax.set_ylabel("Mean Test Error Rate (%)")
    ax.set_title("Error Rate Flow Across Training Artifacts")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    _set_ylim_with_padding(ax)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    return _save_figure(fig, output_dir, "fig3_stage_accuracy_flow", formats)


def plot_onnx_metric_delta(records, output_dir, formats):
    rows = _onnx_delta_rows(records)
    if not rows:
        return []

    grouped = _group_rows(rows, "branch")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=False)
    for branch, branch_rows in grouped.items():
        branch_rows = sorted(
            branch_rows,
            key=lambda row: (natural_key(row["model_name"]), natural_key(row["experiment_name"])),
        )
        x_values = list(range(len(branch_rows)))
        color = "#0072B2" if branch == "pruning_fp16" else "#D55E00"
        error_delta = [row["metric_delta_error_rate"] * 100 for row in branch_rows]
        loss_delta = [row["metric_delta_loss"] for row in branch_rows]
        axes[0].scatter(x_values, error_delta, label=branch, color=color, alpha=0.75)
        axes[1].scatter(x_values, loss_delta, label=branch, color=color, alpha=0.75)

    axes[0].axhline(0, color="#333333", linewidth=1)
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[0].set_ylabel("Error Rate Delta (pp)")
    axes[1].set_ylabel("Loss Delta")
    axes[0].set_title("ONNX Error Rate Delta")
    axes[1].set_title("ONNX Loss Delta")
    for ax in axes:
        ax.set_xlabel("Exported Artifacts")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend()
    return _save_figure(fig, output_dir, "fig4_onnx_metric_delta", formats)


def plot_interface_matrix(records, output_dir, formats):
    rows = _interface_rows(records)
    if not rows:
        return []

    display_rows = rows[:12]
    cell_text = [
        [
            row["stage"],
            row["branch"],
            row["model_name"],
            row["input_dtype"],
            row["input_shape"],
            row["count"],
        ]
        for row in display_rows
    ]
    headers = ["Stage", "Branch", "Model", "Input", "Shape", "Count"]

    fig, ax = plt.subplots(figsize=(12, max(3.6, 0.42 * len(display_rows) + 1.4)))
    ax.axis("off")
    ax.set_title("AMCT / ATC Interface Matrix", pad=18)
    table = ax.table(cellText=cell_text, colLabels=headers, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row_index, _), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#DDEAF6")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#F8F9FA" if row_index % 2 == 0 else "#FFFFFF")
    return _save_figure(fig, output_dir, "fig5_atc_amct_interface_matrix", formats)


def _pruning_rows(records):
    rows = []
    for record in records:
        if record.stage != "pruning":
            continue
        accuracy = _first_number(record.test_acc, record.val_acc)
        error_rate = _error_rate(accuracy)
        if (
            record.pruning_ratio is None
            or accuracy is None
            or record.params is None
            or record.baseline_params in (None, 0)
        ):
            continue
        row = {
            "model_name": record.model_name,
            "experiment_name": record.experiment_name,
            "pruning_ratio": record.pruning_ratio,
            "pruning_steps": record.pruning_steps,
            "accuracy": accuracy,
            "error_rate": error_rate,
            "params": record.params,
            "baseline_params": record.baseline_params,
            "params_remaining_ratio": record.params / record.baseline_params,
            "macs": record.macs,
            "baseline_macs": record.baseline_macs,
            "macs_remaining_ratio": (
                record.macs / record.baseline_macs
                if record.macs is not None and record.baseline_macs not in (None, 0)
                else None
            ),
            "summary_path": record.summary_path,
        }
        rows.append(row)
    return rows


def _stage_error_rows(records):
    stage_order = [
        ("pruning", None, "Pruning"),
        ("qat", None, "QAT"),
        ("onnx", "pruning_fp16", "ONNX FP16"),
        ("onnx", "qat_convert", "ONNX QAT"),
    ]
    rows = []
    for stage, branch, label in stage_order:
        values = []
        for record in records:
            if record.stage != stage:
                continue
            if branch is not None and record.branch != branch:
                continue
            accuracy = _first_number(record.exported_test_acc, record.test_acc, record.val_acc)
            error_rate = _error_rate(accuracy)
            if error_rate is not None:
                values.append(error_rate)
        if values:
            rows.append(
                {
                    "stage": stage,
                    "branch": branch,
                    "stage_label": label,
                    "count": len(values),
                    "mean_error_rate": sum(values) / len(values),
                    "min_error_rate": min(values),
                    "max_error_rate": max(values),
                }
            )
    return rows


def _onnx_delta_rows(records):
    rows = []
    for record in records:
        if record.stage != "onnx":
            continue
        if record.metric_delta_acc is None or record.metric_delta_loss is None:
            continue
        source_error_rate = _error_rate(record.source_test_acc)
        exported_error_rate = _error_rate(record.exported_test_acc)
        rows.append(
            {
                "branch": record.branch,
                "model_name": record.model_name,
                "experiment_name": record.experiment_name,
                "metric_delta_acc": record.metric_delta_acc,
                "metric_delta_error_rate": -record.metric_delta_acc,
                "metric_delta_loss": record.metric_delta_loss,
                "source_test_acc": record.source_test_acc,
                "exported_test_acc": record.exported_test_acc,
                "source_test_error_rate": source_error_rate,
                "exported_test_error_rate": exported_error_rate,
                "summary_path": record.summary_path,
            }
        )
    return rows


def _interface_rows(records):
    grouped = defaultdict(int)
    for record in records:
        if record.stage not in {"amct", "atc"}:
            continue
        if not record.input_shape:
            continue
        key = (
            record.stage,
            record.branch or "",
            record.model_name,
            record.input_dtype or str(record.input_elem_type or ""),
            record.output_dtype or str(record.output_elem_type or ""),
            record.input_shape,
            record.output_shape or "",
            record.soc_version or "",
        )
        grouped[key] += 1

    rows = []
    for key, count in grouped.items():
        stage, branch, model_name, input_dtype, output_dtype, input_shape, output_shape, soc_version = key
        rows.append(
            {
                "stage": stage,
                "branch": branch,
                "model_name": model_name,
                "input_dtype": input_dtype,
                "output_dtype": output_dtype,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "soc_version": soc_version,
                "count": count,
            }
        )
    rows.sort(
        key=lambda row: (
            row["stage"],
            row["branch"],
            natural_key(row["model_name"]),
            row["input_dtype"],
        )
    )
    return rows


def _save_figure(fig, output_dir, stem, formats):
    output_dir = Path(output_dir)
    fig.tight_layout()
    paths = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        save_kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            save_kwargs["dpi"] = 200
        fig.savefig(path, **save_kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def _group_records(records, attr_name):
    grouped = defaultdict(list)
    for record in records:
        grouped[getattr(record, attr_name)].append(record)
    return grouped


def _group_rows(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return grouped


def _color_for_model(model_name):
    return MODEL_COLORS.get(model_name, "#666666")


def _paired_bars(ax, x_values, baseline, final, labels, width, ylabel, title, yscale=None):
    ax.bar([x - width / 2 for x in x_values], baseline, width, label="Baseline", color="#7AA6C2")
    ax.bar([x + width / 2 for x in x_values], final, width, label="Final", color="#D7835F")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend()


def _set_ylim_with_padding(ax):
    bottom, top = ax.get_ylim()
    if top <= bottom:
        return
    padding = (top - bottom) * 0.08
    ax.set_ylim(bottom, top + padding)


def _first_number(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _error_rate(accuracy):
    if accuracy is None:
        return None
    error_rate = 1.0 - accuracy
    if error_rate < 0:
        return None
    return error_rate


def _is_positive(value):
    return value is not None and value > 0
