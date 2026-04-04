"""Plotting utilities for evaluation results and training metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def set_style():
    """Set consistent plot style."""
    sns.set_theme(style="whitegrid", palette="husl")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def plot_per_class_ap(
    per_class_ap: dict[str, float],
    output_path: Path,
    title: str = "Per-Class Average Precision (AP@50)",
) -> Path:
    """Create a horizontal bar chart of per-class AP values."""
    set_style()

    sorted_items = sorted(per_class_ap.items(), key=lambda x: x[1])
    classes = [item[0] for item in sorted_items]
    aps = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(4, len(classes) * 0.4)))
    colors = sns.color_palette("viridis", len(classes))
    bars = ax.barh(classes, aps, color=colors)

    # Add value labels
    for bar, ap in zip(bars, aps):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{ap:.3f}", va="center", fontsize=9)

    ax.set_xlabel("Average Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1.1)
    ax.axvline(x=np.mean(aps), color="red", linestyle="--", alpha=0.7, label=f"mAP: {np.mean(aps):.3f}")
    ax.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_precision_recall_curve(
    precisions: dict[str, np.ndarray],
    recalls: dict[str, np.ndarray],
    output_path: Path,
    title: str = "Precision-Recall Curves",
) -> Path:
    """Plot precision-recall curves for multiple classes."""
    set_style()

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("husl", len(precisions))

    for (cls_name, prec), (_, rec), color in zip(
        precisions.items(), recalls.items(), colors
    ):
        ax.plot(rec, prec, color=color, label=cls_name, linewidth=1.5)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> Path:
    """Plot a confusion matrix heatmap."""
    set_style()

    labels = class_names + ["background"]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), max(6, len(labels) * 0.5)))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_training_loss(
    log_path: Path,
    output_path: Path,
    title: str = "Training Loss",
) -> Path:
    """Plot training loss curve from a training log JSON."""
    set_style()

    with open(log_path) as f:
        log_data = json.load(f)

    epochs = list(range(1, len(log_data) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    if isinstance(log_data, list):
        losses = [entry.get("loss", entry.get("train_loss", 0)) for entry in log_data]
        axes[0].plot(epochs, losses, "b-", linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")

        # mAP curve if available
        maps = [entry.get("mAP_50_95", entry.get("mAP", None)) for entry in log_data]
        if any(m is not None for m in maps):
            maps = [m or 0 for m in maps]
            axes[1].plot(epochs, maps, "g-", linewidth=2)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("mAP@50:95")
            axes[1].set_title("Validation mAP")
    else:
        axes[0].text(0.5, 0.5, "No training data available", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No validation data available", ha="center", va="center")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_model_comparison(
    results: dict[str, dict],
    output_path: Path,
    title: str = "Model Comparison",
) -> Path:
    """Create a grouped bar chart comparing multiple model runs.

    Args:
        results: Dict mapping run_name -> {metric_name: value}.
        output_path: Where to save the plot.
    """
    set_style()

    metrics = ["mAP_50_95", "mAP_50", "mAP_75"]
    metric_labels = ["mAP@50:95", "mAP@50", "mAP@75"]
    run_names = list(results.keys())

    x = np.arange(len(metrics))
    width = 0.8 / len(run_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("husl", len(run_names))

    for i, (run_name, run_metrics) in enumerate(results.items()):
        values = [run_metrics.get(m, 0) for m in metrics]
        offset = (i - len(run_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=run_name, color=colors[i])

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1.15)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
