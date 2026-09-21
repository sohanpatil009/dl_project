"""Central configuration for the Wideband Signal Recognition project.

All hyperparameters, paths and constants used across the project are
defined here so experiments can be reproduced exactly. A simple YAML
file (`configs/default.yaml`) is loaded on top of these defaults.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


def _project_root() -> Path:
    """Return the project root regardless of where Python is invoked."""
    return PROJECT_ROOT


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
DEFAULT_SEED: int = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy and PyTorch (CPU + GPU) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms are slower; the user can opt in.
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Paper-aligned dataset parameters
# ---------------------------------------------------------------------------
# The original Vagollari et al. (2023) paper does not release its synthetic
# wideband dataset. The values below reproduce the *parameters* described
# in the paper as closely as possible.

# Wideband capture
WIDEBAND_SAMPLING_RATE_HZ: float = 200_000.0  # 200 kHz, paper Section III
WIDEBAND_DURATION_S: float = 0.5              # capture length in seconds
WIDEBAND_CARRIER_HZ: float = 0.0              # baseband, paper-style

# Spectrogram
SPECTROGRAM_FFT_SIZE: int = 1024              # paper Section III-B
SPECTROGRAM_HOP_LENGTH: int = 64              # ~93.75% overlap
SPECTROGRAM_WINDOW: str = "hann"              # paper uses Hann
SPECTROGRAM_DB_FLOOR: float = -120.0          # dynamic range floor
SPECTROGRAM_DB_CEIL: float = 0.0
SPECTROGRAM_IMAGE_SIZE: int = 1024            # width fed to YOLO
SPECTROGRAM_COLORMAP: str = "viridis"         # informational, not for training

# Per-narrowband signal generation
NARROWBAND_SYMBOL_RATE_HZ: float = 10_000.0   # 10 ksym/s
NARROWBAND_ROLL_OFF: float = 0.35
NARROWBAND_MAX_PER_WIDEBAND: int = 4          # paper "up to 4" signals
NARROWBAND_MIN_PER_WIDEBAND: int = 1
NARROWBAND_BANDWIDTH_HZ: float = 12_000.0     # symbol_rate * (1 + rolloff)
NARROWBAND_PADDING_HZ: float = 2_000.0        # half-bandwidth padding for bbox

# Modulation classes (paper Section III-C)
MODULATION_CLASSES: List[str] = [
    "BPSK",
    "QPSK",
    "8PSK",
    "16QAM",
    "64QAM",
    "NOISE",
    "NO_SIGNAL",
]
NUM_CLASSES: int = len(MODULATION_CLASSES)
# Index used for empty regions (background in YOLO terms)
NO_SIGNAL_INDEX: int = MODULATION_CLASSES.index("NO_SIGNAL")

# SNR range (paper trains at 20 dB, evaluates 0-30 dB)
SNR_TRAIN_DB: float = 20.0
SNR_EVAL_DB_LIST: List[float] = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]

# Dataset sizes
WIDEBAND_TRAIN: int = 600
WIDEBAND_VAL: int = 150
WIDEBAND_TEST: int = 150
# Narrowband clips per modulation (used for the standalone classifier)
NARROWBAND_PER_CLASS: int = 400

# Data source: "synthetic" (built-in generator) or "radioml" (real internet
# data, RadioML 2016.10A, mixed into wideband captures as documented in
# src/public_data.py).
DATA_SOURCE: str = "synthetic"
# Spectrogram image container: "jpg" (smallest), "png" (lossless) or "bmp".
IMAGE_FORMAT: str = "jpg"
# RadioML options (only used when data_source == "radioml")
RADIOML_URL: str = "https://zenodo.org/api/records/18397070/files/RML2016.10a.tar.bz2/content"
RADIOML_SNR_DB: float = 18.0   # cleanest RadioML SNR bin near the paper's 20 dB train point
RADIOML_INCLUDE_EXTRA: bool = False  # also keep CPFSK/GFSK/PAM4/AM-*/WBFM (non-paper classes)

# ---------------------------------------------------------------------------
# YOLOv5 detector
# ---------------------------------------------------------------------------
YOLO_INPUT_SIZE: int = 640
YOLO_BATCH_SIZE: int = 16
YOLO_EPOCHS: int = 30
YOLO_PATIENCE: int = 10
YOLO_LEARNING_RATE: float = 1e-3
YOLO_WEIGHT_DECAY: float = 5e-4
YOLO_CONF_THRESHOLD: float = 0.25
YOLO_IOU_THRESHOLD: float = 0.45
YOLO_VARIANT: str = "yolov5s"  # YOLOv5 small, as in the paper

# ---------------------------------------------------------------------------
# Modulation classifier
# ---------------------------------------------------------------------------
CLASSIFIER_SEQ_LEN: int = 128                 # IQ samples per example
CLASSIFIER_BATCH_SIZE: int = 128
CLASSIFIER_EPOCHS: int = 40
CLASSIFIER_PATIENCE: int = 8
CLASSIFIER_LEARNING_RATE: float = 1e-3
CLASSIFIER_WEIGHT_DECAY: float = 1e-4
CLASSIFIER_ARCH: str = "resnet18"             # {"cnn1d", "resnet18", "resnet34"}
CLASSIFIER_USE_TRANSFER: bool = True          # load ImageNet weights for ResNet
CLASSIFIER_USE_AUGMENTATION: bool = True

# ---------------------------------------------------------------------------
# Aggregate config dataclass
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Container holding every project parameter."""

    seed: int = DEFAULT_SEED

    # wideband / spectrogram
    fs: float = WIDEBAND_SAMPLING_RATE_HZ
    duration: float = WIDEBAND_DURATION_S
    n_fft: int = SPECTROGRAM_FFT_SIZE
    hop_length: int = SPECTROGRAM_HOP_LENGTH
    window: str = SPECTROGRAM_WINDOW
    db_floor: float = SPECTROGRAM_DB_FLOOR
    image_size: int = SPECTROGRAM_IMAGE_SIZE

    # narrowband signal
    symbol_rate: float = NARROWBAND_SYMBOL_RATE_HZ
    roll_off: float = NARROWBAND_ROLL_OFF
    max_signals: int = NARROWBAND_MAX_PER_WIDEBAND
    min_signals: int = NARROWBAND_MIN_PER_WIDEBAND
    bandwidth: float = NARROWBAND_BANDWIDTH_HZ
    padding: float = NARROWBAND_PADDING_HZ

    # classes
    mod_classes: List[str] = field(default_factory=lambda: list(MODULATION_CLASSES))
    num_classes: int = NUM_CLASSES
    no_signal_index: int = NO_SIGNAL_INDEX

    # SNR
    snr_train_db: float = SNR_TRAIN_DB
    snr_eval_db: List[float] = field(default_factory=lambda: list(SNR_EVAL_DB_LIST))

    # dataset sizes
    wideband_train: int = WIDEBAND_TRAIN
    wideband_val: int = WIDEBAND_VAL
    wideband_test: int = WIDEBAND_TEST
    narrowband_per_class: int = NARROWBAND_PER_CLASS

    # data source / image container / public-data options
    data_source: str = DATA_SOURCE
    image_format: str = IMAGE_FORMAT
    radioml_url: str = RADIOML_URL
    radioml_snr_db: float = RADIOML_SNR_DB
    radioml_include_extra: bool = RADIOML_INCLUDE_EXTRA

    # yolo
    yolo_input_size: int = YOLO_INPUT_SIZE
    yolo_batch_size: int = YOLO_BATCH_SIZE
    yolo_epochs: int = YOLO_EPOCHS
    yolo_patience: int = YOLO_PATIENCE
    yolo_lr: float = YOLO_LEARNING_RATE
    yolo_weight_decay: float = YOLO_WEIGHT_DECAY
    yolo_conf: float = YOLO_CONF_THRESHOLD
    yolo_iou: float = YOLO_IOU_THRESHOLD
    yolo_variant: str = YOLO_VARIANT

    # classifier
    clf_seq_len: int = CLASSIFIER_SEQ_LEN
    clf_batch_size: int = CLASSIFIER_BATCH_SIZE
    clf_epochs: int = CLASSIFIER_EPOCHS
    clf_patience: int = CLASSIFIER_PATIENCE
    clf_lr: float = CLASSIFIER_LEARNING_RATE
    clf_weight_decay: float = CLASSIFIER_WEIGHT_DECAY
    clf_arch: str = CLASSIFIER_ARCH
    clf_use_transfer: bool = CLASSIFIER_USE_TRANSFER
    clf_use_augmentation: bool = CLASSIFIER_USE_AUGMENTATION

    # paths
    project_root: str = str(PROJECT_ROOT)
    data_dir: str = str(DATA_DIR)
    models_dir: str = str(MODELS_DIR)
    results_dir: str = str(RESULTS_DIR)
    figures_dir: str = str(FIGURES_DIR)
    metrics_dir: str = str(METRICS_DIR)
    predictions_dir: str = str(PREDICTIONS_DIR)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Optional[str] = None) -> str:
        path = path or str(METRICS_DIR / "config.yaml")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
        return path

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__})


def get_default_config() -> Config:
    """Return a fresh Config populated with paper defaults."""
    return Config()


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return best available torch device (CUDA > MPS > CPU)."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_cuda and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dirs(*paths: str) -> None:
    """Create the given directories if they do not exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
def _print_config(cfg: Config) -> None:
    import json
    print(json.dumps(cfg.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    cfg = get_default_config()
    set_seed(cfg.seed)
    _print_config(cfg)
