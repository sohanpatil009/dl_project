"""Dataset utilities.

This module contains file-level helpers used both to (a) persist
synthetic wideband captures to disk and (b) convert them to YOLO-style
labels for the detector stage.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .preprocessing import NarrowbandSignal, WidebandSample
from .spectrogram import Spectrogram, compute_spectrogram, spectrogram_to_image


# ---------------------------------------------------------------------------
# File I/O for synthetic captures
# ---------------------------------------------------------------------------
@dataclass
class StoredSample:
    """Metadata stored alongside a spectrogram on disk."""

    sample_id: int
    spec_path: str
    label_path: Optional[str]
    n_signals: int
    snr_db: float
    signals: List[Dict[str, float]]


IMAGE_FORMATS: Dict[str, Tuple[str, str]] = {
    # ext -> (pillow format, filename ext)
    "jpg": ("JPEG", "jpg"),
    "jpeg": ("JPEG", "jpg"),
    "png": ("PNG", "png"),
    "bmp": ("BMP", "bmp"),
}


def save_wideband_dataset(samples: List[WidebandSample], out_dir: str,
                          spec_image_size: int = 1024,
                          n_fft: int = 1024, hop_length: int = 64,
                          window: str = "hann",
                          db_floor: float = -120.0,
                          make_yolo_labels: bool = True,
                          yolo_class_map: Optional[Dict[str, int]] = None,
                          bandwidth_hz: float = 12_000.0,
                          padding_hz: float = 2_000.0,
                          num_classes: int = 6,
                          image_format: str = "jpg") -> List[StoredSample]:
    """Save a list of WidebandSample as spectrogram images + YOLO labels.

    Parameters
    ----------
    samples : list
        Output of `generate_wideband` (or a custom generator).
    out_dir : str
        Output directory.
    spec_image_size : int
        Side length (px) of the saved image.
    n_fft, hop_length, window, db_floor
        STFT parameters.
    make_yolo_labels : bool
        If True, also write a YOLO-format ``.txt`` file per image.
    yolo_class_map : dict
        Maps modulation name -> YOLO class index. If None, every signal
        is treated as class 0 (a single "signal" class) which is the
        approach used in the paper. The classifier then provides the
        modulation label. Set this to a per-modulation mapping if you
        want YOLO to predict the modulation directly.
    bandwidth_hz, padding_hz : float
        Bounding-box width in frequency (Hz).
    num_classes : int
        Number of classes for the YOLO YAML config.
    image_format : str
        One of {"jpg", "png", "bmp"} (jpg = smallest, png = lossless).

    Returns
    -------
    list of StoredSample
    """
    fmt_key = image_format.lower().lstrip(".")
    if fmt_key not in IMAGE_FORMATS:
        raise ValueError(f"Unsupported image_format {image_format!r}; "
                         f"choose from {sorted(IMAGE_FORMATS)}.")
    pil_format, ext = IMAGE_FORMATS[fmt_key]
    img_dir = Path(out_dir) / "images"
    lbl_dir = Path(out_dir) / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    if make_yolo_labels:
        lbl_dir.mkdir(parents=True, exist_ok=True)
    stored: List[StoredSample] = []
    for idx, s in enumerate(samples):
        spec = compute_spectrogram(s.samples, fs=s.fs, n_fft=n_fft,
                                   hop_length=hop_length, window=window,
                                   db_floor=db_floor)
        img = spectrogram_to_image(spec, image_size=spec_image_size)
        spec_path = img_dir / f"sample_{idx:06d}.{ext}"
        # Save with PIL to keep file size reasonable
        from PIL import Image
        save_kwargs = {"format": pil_format}
        if pil_format == "JPEG":
            save_kwargs["quality"] = 90
        Image.fromarray(img, mode="RGB").save(spec_path, **save_kwargs)

        label_path: Optional[str] = None
        if make_yolo_labels:
            label_path = str(lbl_dir / f"sample_{idx:06d}.txt")
            _write_yolo_label(
                label_path=label_path,
                spec=spec,
                signals=s.signals,
                image_size=spec_image_size,
                yolo_class_map=yolo_class_map,
                bandwidth_hz=bandwidth_hz,
                padding_hz=padding_hz,
            )
        stored.append(StoredSample(
            sample_id=idx,
            spec_path=str(spec_path),
            label_path=label_path,
            n_signals=len(s.signals),
            snr_db=s.snr_db,
            signals=[
                {"modulation": nbs.modulation, "carrier_offset_hz": nbs.carrier_offset_hz,
                 "start_sample": nbs.start_sample, "snr_db": nbs.snr_db}
                for nbs in s.signals
            ],
        ))
    # Save an index.json with metadata for later inspection
    with open(Path(out_dir) / "index.json", "w", encoding="utf-8") as f:
        json.dump([s.__dict__ for s in stored], f, indent=2)
    return stored


# ---------------------------------------------------------------------------
# YOLO label writing
# ---------------------------------------------------------------------------
def _write_yolo_label(label_path: str, spec: Spectrogram, signals: Sequence[NarrowbandSignal],
                      image_size: int, yolo_class_map: Optional[Dict[str, int]],
                      bandwidth_hz: float, padding_hz: float) -> None:
    """Write a YOLO .txt file.

    The bounding box covers the signal bandwidth plus a small padding to
    emulate the "wide" boxes used in the paper.
    All coordinates are in YOLO format: ``cls cx cy w h`` (normalised
    0..1). The single-channel "signal" class is used unless
    ``yolo_class_map`` is provided.
    """
    n_freq, n_time = spec.shape
    f_lo, f_hi = float(spec.freq_hz.min()), float(spec.freq_hz.max())
    t_lo, t_hi = float(spec.time_s.min()), float(spec.time_s.max())
    bw = bandwidth_hz + 2 * padding_hz
    lines: List[str] = []
    for nbs in signals:
        cf = nbs.carrier_offset_hz
        cx_freq = cf
        cy_time = 0.5 * (t_lo + t_hi)
        w_freq = bw
        h_time = (t_hi - t_lo)
        # Normalise to 0..1
        cx_n = (cx_freq - f_lo) / (f_hi - f_lo + 1e-12)
        cy_n = (cy_time - t_lo) / (t_hi - t_lo + 1e-12)
        w_n = w_freq / (f_hi - f_lo + 1e-12)
        h_n = h_time / (t_hi - t_lo + 1e-12)
        cx_n = float(np.clip(cx_n, 0.0, 1.0))
        cy_n = float(np.clip(cy_n, 0.0, 1.0))
        w_n = float(np.clip(w_n, 0.0, 1.0))
        h_n = float(np.clip(h_n, 0.0, 1.0))
        if yolo_class_map is None:
            cls = 0
        else:
            cls = int(yolo_class_map.get(nbs.modulation, 0))
        lines.append(f"{cls} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_yolo_yaml(out_path: str, dataset_dir: str,
                    num_classes: int = 1,
                    class_names: Optional[Sequence[str]] = None) -> str:
    """Write the YOLO dataset configuration file (data.yaml)."""
    if class_names is None:
        class_names = ["signal"] if num_classes == 1 else [f"class_{i}" for i in range(num_classes)]
    cfg = {
        "path": str(Path(dataset_dir).resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": num_classes,
        "names": list(class_names),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import yaml
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def split_yolo_dataset(stored: List[StoredSample], out_dir: str,
                       train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, List[str]]:
    """Move/copy the saved images+labels into train/val/test folders."""
    out_dir = Path(out_dir)
    img_root = out_dir / "images"
    lbl_root = out_dir / "labels"
    for sub in ("train", "val", "test"):
        (img_root / sub).mkdir(parents=True, exist_ok=True)
        (lbl_root / sub).mkdir(parents=True, exist_ok=True)
    n = len(stored)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    train_idx = set(idx[:n_train].tolist())
    val_idx = set(idx[n_train:n_train + n_val].tolist())
    test_idx = set(idx[n_train + n_val:].tolist())
    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for s in stored:
        sub = "train" if s.sample_id in train_idx else "val" if s.sample_id in val_idx else "test"
        src_img = Path(s.spec_path)
        src_lbl = Path(s.label_path) if s.label_path else None
        dst_img = img_root / sub / src_img.name
        dst_lbl = lbl_root / sub / src_lbl.name if src_lbl else None
        if src_img.resolve() != dst_img.resolve():
            shutil.copy2(src_img, dst_img)
        if src_lbl and src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)
        splits[sub].append(str(dst_img))
    return splits


# ---------------------------------------------------------------------------
# Narrowband clip persistence (for the classifier)
# ---------------------------------------------------------------------------
def save_narrowband_clips(samples_by_class: Dict[str, List[np.ndarray]], out_dir: str,
                          sample_len: int) -> List[Tuple[str, str]]:
    """Save a dict {class_name: list of complex arrays} as .npy files.

    Returns
    -------
    list of (class, path) tuples that form the manifest.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: List[Tuple[str, str]] = []
    for cls, arrs in samples_by_class.items():
        cls_dir = out / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i, arr in enumerate(arrs):
            # Trim / pad to fixed length
            if len(arr) > sample_len:
                start = (len(arr) - sample_len) // 2
                arr = arr[start:start + sample_len]
            elif len(arr) < sample_len:
                arr = np.pad(arr, (0, sample_len - len(arr)))
            p = cls_dir / f"{cls}_{i:06d}.npy"
            np.save(p, arr.astype(np.complex64))
            manifest.append((cls, str(p)))
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return manifest
