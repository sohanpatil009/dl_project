"""Publication-quality plotting helpers.

Every figure produced by the project uses these helpers so that styles
are consistent across notebooks and saved figures.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib
# Use a headless backend only when NOT running inside a notebook/Colab.
# Forcing "Agg" unconditionally breaks plt.show() in Colab/Jupyter
# (figures render as blank). In scripts (no DISPLAY) keep Agg.
import os as _os
if _os.environ.get("MPLBACKEND") is None:
    try:
        from IPython import get_ipython as _get_ipython  # type: ignore
        _ip = _get_ipython()
        _in_notebook = _ip is not None and getattr(_ip, "kernel", None) is not None
    except Exception:
        _in_notebook = False
    if not _in_notebook and not _os.environ.get("DISPLAY"):
        matplotlib.use("Agg")  # headless backend for plain scripts
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# RF signal / spectrogram
# ---------------------------------------------------------------------------
def plot_iq(samples: np.ndarray, fs: float, title: str = "I/Q signal",
            n: int = 1024, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 3))
    t = np.arange(min(n, len(samples))) / fs
    s = samples[:n]
    ax.plot(t * 1e3, np.real(s), label="I", linewidth=0.8)
    ax.plot(t * 1e3, np.imag(s), label="Q", linewidth=0.8, alpha=0.8)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.3)
    return ax


def plot_spectrogram(spec_db: np.ndarray, freq_hz: np.ndarray, time_s: np.ndarray,
                     title: str = "Spectrogram", ax: Optional[plt.Axes] = None,
                     cmap: str = "viridis") -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(7, 5))
    extent = [time_s[0] * 1e3, time_s[-1] * 1e3,
              freq_hz[0] / 1e3, freq_hz[-1] / 1e3]
    im = ax.imshow(spec_db, origin="lower", aspect="auto", cmap=cmap, extent=extent)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Frequency [kHz]")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Power [dB]")
    return ax


def draw_boxes(ax: plt.Axes, boxes: Sequence[Tuple[float, float, float, float]],
               labels: Optional[Sequence[str]] = None,
               scores: Optional[Sequence[float]] = None,
               color: str = "red", line_width: float = 2.0) -> None:
    """Draw YOLO-style boxes on a spectrogram axes.

    `boxes` are ``(cx, cy, w, h)`` in 0..1 (YOLO format) or in axis data
    units (depending on the caller's intent). The axes' x and y limits
    are used to scale accordingly.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    W = x1 - x0
    H = y1 - y0
    for i, (cx, cy, w, h) in enumerate(boxes):
        if 0.0 <= cx <= 1.0 and 0.0 <= w <= 1.0:
            cx_ax = x0 + cx * W
            cy_ax = y0 + cy * H
            w_ax = w * W
            h_ax = h * H
        else:
            cx_ax, cy_ax, w_ax, h_ax = cx, cy, w, h
        rect = mpatches.Rectangle(
            (cx_ax - w_ax / 2, cy_ax - h_ax / 2), w_ax, h_ax,
            fill=False, edgecolor=color, linewidth=line_width,
        )
        ax.add_patch(rect)
        if labels is not None:
            txt = labels[i] if i < len(labels) else ""
            if scores is not None and i < len(scores):
                txt = f"{txt} {scores[i]:.2f}"
            ax.text(cx_ax - w_ax / 2, cy_ax + h_ax / 2, txt,
                    color=color, fontsize=8, va="bottom")


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def plot_training_curves(history: dict, out_path: Optional[str] = None) -> plt.Figure:
    """Plot loss/accuracy curves from a `history` dict."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Loss
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="train")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # Accuracy
    if "train_acc" in history:
        axes[1].plot(history["train_acc"], label="train")
    if "val_acc" in history:
        axes[1].plot(history["val_acc"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        _ensure_dir(os.path.dirname(out_path))
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str],
                          normalize: bool = True,
                          out_path: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.9),
                                   max(5, len(class_names) * 0.8)))
    cm_show = cm.astype(float)
    if normalize:
        cm_show = cm_show / (cm_show.sum(axis=1, keepdims=True) + 1e-12)
    im = ax.imshow(cm_show, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (normalised)" if normalize else ""))
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_show[i, j]
            ax.text(j, i, f"{val:.2f}" if normalize else f"{int(val)}",
                    ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if out_path:
        _ensure_dir(os.path.dirname(out_path))
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Performance vs SNR
# ---------------------------------------------------------------------------
def plot_metric_vs_snr(snr: Sequence[float], metric_dict: dict, ylabel: str = "Metric",
                        out_path: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, values in metric_dict.items():
        ax.plot(snr, values, marker="o", label=name)
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs SNR")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if out_path:
        _ensure_dir(os.path.dirname(out_path))
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pr_curve(recall: Sequence[float], precision: Sequence[float],
                  out_path: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="C1", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        _ensure_dir(os.path.dirname(out_path))
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
