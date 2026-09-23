"""YOLOv5 utilities.

We do *not* ship a YOLOv5 fork inside this repository. Instead, this
module clones the official Ultralytics YOLOv5 repository at runtime
(when needed) into ``models/yolov5/`` and exposes a small wrapper that
is used by the rest of the project. The wrapper:

* prepares the dataset configuration,
* triggers training,
* loads a trained model,
* runs inference on a list of images / spectrograms,
* post-processes the output into a Python-friendly list of detections.

If the clone fails (e.g. no internet), the user is informed clearly
and the alternative is to use any compatible ``.pt`` checkpoint.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from .config import Config, get_default_config, get_device, PROJECT_ROOT


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
YOLOV5_DIR: Path = PROJECT_ROOT / "models" / "yolov5"


def yolov5_dir() -> Path:
    return YOLOV5_DIR


def yolov5_available() -> bool:
    """Return True if the yolov5 repo + its requirements are present."""
    return (YOLOV5_DIR / "train.py").is_file() and (YOLOV5_DIR / "detect.py").is_file()


def install_yolov5(force: bool = False) -> Path:
    """Ensure the Ultralytics runtime is importable.

    Historically this cloned the YOLOv5 repo; current training uses the
    ``ultralytics`` package directly, so we only ensure it is installed.
    On Colab torch is preinstalled with CUDA — we must NOT pull
    extra heavy deps (roboflow/albumentations) that force numpy/torch
    downgrades, so install is limited to ``ultralytics``.
    Returns the path to the cloned repository.
    """
    if yolov5_available() and not force:
        return YOLOV5_DIR
    if YOLOV5_DIR.exists() and force:
        shutil.rmtree(YOLOV5_DIR, ignore_errors=True)
    YOLOV5_DIR.parent.mkdir(parents=True, exist_ok=True)
    try:
        import ultralytics  # noqa: F401  type: ignore
        return YOLOV5_DIR
    except ImportError:
        pass
    cmd = [
        sys.executable, "-m", "pip", "install", "-q",
        "ultralytics>=8.0.0",
    ]
    print("[yolov5] Installing Ultralytics runtime (ultralytics only) ...")
    try:
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT, timeout=300)
    except Exception as exc:  # noqa: BLE001
        print(f"[yolov5] WARNING: pip install failed: {exc}")
    # Use Ultralytics package (which provides YOLOv5 weights) directly
    return YOLOV5_DIR


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """Single detection returned by the YOLO wrapper."""

    image_path: str
    class_id: int
    class_name: str
    confidence: float
    cx: float  # 0..1
    cy: float  # 0..1
    w: float   # 0..1
    h: float   # 0..1

    def to_xyxy(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        x1 = (self.cx - self.w / 2) * img_w
        y1 = (self.cy - self.h / 2) * img_h
        x2 = (self.cx + self.w / 2) * img_w
        y2 = (self.cy + self.h / 2) * img_h
        return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_yolov5(data_yaml: str, cfg: Optional[Config] = None,
                 epochs: Optional[int] = None, batch: Optional[int] = None,
                 img_size: Optional[int] = None,
                 weights: str = "yolov5s.pt",
                 project: str = "models/runs",
                 name: str = "detector",
                 device: str = "",
                 patience: Optional[int] = None,
                 exist_ok: bool = True) -> Path:
    """Train YOLOv5 with the Ultralytics runtime.

    This is the most reliable way to train YOLOv5 in 2024+ because the
    legacy `train.py` requires old torch / albumentations versions.

    Returns the path to ``best.pt``.
    """
    if cfg is None:
        cfg = get_default_config()
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Ultralytics is not installed. Run `pip install ultralytics` "
            "or call `install_yolov5()` first."
        ) from e
    epochs = epochs if epochs is not None else cfg.yolo_epochs
    batch = batch if batch is not None else cfg.yolo_batch_size
    img_size = img_size if img_size is not None else cfg.yolo_input_size
    patience = patience if patience is not None else cfg.yolo_patience
    if not device:
        device = "0" if get_device().type == "cuda" else "cpu"
    # Absolute project dir: newer Ultralytics nests *relative* project paths
    # inside its own runs/ folder, which would hide the checkpoints from us.
    project = str(Path(project).resolve())
    print(f"[yolov5] Training: data={data_yaml} epochs={epochs} batch={batch} "
          f"img={img_size} device={device} weights={weights}")
    model = YOLO(weights)
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        patience=patience,
        exist_ok=exist_ok,
        lr0=cfg.yolo_lr,
        weight_decay=cfg.yolo_weight_decay,
        verbose=True,
    )
    # Prefer the trainer's real save dir (robust to Ultralytics path logic)
    try:
        run_dir = Path(model.trainer.save_dir)
    except Exception:  # noqa: BLE001
        run_dir = Path(project) / name
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        # Fallback to last
        best = run_dir / "weights" / "last.pt"
    if not best.exists():
        raise RuntimeError(
            f"YOLO training finished but no checkpoint found in "
            f"{run_dir / 'weights'}. Check the Ultralytics log above.")
    return best


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def load_yolov5(weights: str) -> "YOLO":  # type: ignore[name-defined]
    from ultralytics import YOLO  # type: ignore
    return YOLO(weights)


def run_yolov5_inference(model, images: Sequence[str], cfg: Optional[Config] = None,
                         conf: Optional[float] = None,
                         iou: Optional[float] = None,
                         imgsz: Optional[int] = None,
                         device: str = "",
                         save: bool = False,
                         project: str = "models/runs",
                         name: str = "predict",
                         ) -> List[List[Detection]]:
    """Run YOLOv5 on a list of image paths and return structured detections."""
    if cfg is None:
        cfg = get_default_config()
    conf = conf if conf is not None else cfg.yolo_conf
    iou = iou if iou is not None else cfg.yolo_iou
    imgsz = imgsz if imgsz is not None else cfg.yolo_input_size
    if not device:
        device = "0" if get_device().type == "cuda" else "cpu"
    results = model.predict(
        source=list(images),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        save=save,
        project=project,
        name=name,
        verbose=False,
    )
    class_names = _read_class_names(model)
    out: List[List[Detection]] = []
    for r in results:
        detections: List[Detection] = []
        names = r.names if hasattr(r, "names") else class_names
        if r.boxes is None or len(r.boxes) == 0:
            out.append(detections)
            continue
        xywhn = r.boxes.xywhn.detach().cpu().numpy()
        cls = r.boxes.cls.detach().cpu().numpy().astype(int)
        confs = r.boxes.conf.detach().cpu().numpy()
        for i in range(len(cls)):
            cx, cy, w, h = xywhn[i]
            detections.append(Detection(
                image_path=str(r.path) if hasattr(r, "path") else "",
                class_id=int(cls[i]),
                class_name=names.get(int(cls[i]), str(int(cls[i]))),
                confidence=float(confs[i]),
                cx=float(cx), cy=float(cy), w=float(w), h=float(h),
            ))
        out.append(detections)
    return out


def _read_class_names(model) -> dict:
    """Read class names from a YOLO model. Works with Ultralytics YOLO."""
    try:
        # Ultralytics models expose a `.names` dict after training
        return dict(model.names)
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# YOLO <-> spectrogram coordinate conversions
# ---------------------------------------------------------------------------
def yolo_box_to_freq_time(cx: float, cy: float, w: float, h: float,
                          spec) -> Tuple[float, float, float, float]:
    """Convert YOLO normalised coords to (f_lo, f_hi, t_lo, t_hi) in Hz/s.

    `spec` must be a `Spectrogram` instance providing `freq_hz` and
    `time_s` arrays.
    """
    f_lo = float(spec.freq_hz.min())
    f_hi = float(spec.freq_hz.max())
    t_lo = float(spec.time_s.min())
    t_hi = float(spec.time_s.max())
    center_f = f_lo + cx * (f_hi - f_lo)
    center_t = t_lo + cy * (t_hi - t_lo)
    bw_f = w * (f_hi - f_lo)
    bw_t = h * (t_hi - t_lo)
    return (center_f - bw_f / 2, center_f + bw_f / 2,
            max(0.0, center_t - bw_t / 2), center_t + bw_t / 2)


def freq_time_to_yolo_box(f_lo: float, f_hi: float, t_lo: float, t_hi: float,
                          spec) -> Tuple[float, float, float, float]:
    f_min = float(spec.freq_hz.min())
    f_max = float(spec.freq_hz.max())
    t_min = float(spec.time_s.min())
    t_max = float(spec.time_s.max())
    cx = ((f_lo + f_hi) / 2 - f_min) / (f_max - f_min + 1e-12)
    cy = ((t_lo + t_hi) / 2 - t_min) / (t_max - t_min + 1e-12)
    w = (f_hi - f_lo) / (f_max - f_min + 1e-12)
    h = (t_hi - t_lo) / (t_max - t_min + 1e-12)
    return float(np.clip(cx, 0, 1)), float(np.clip(cy, 0, 1)), \
        float(np.clip(w, 0, 1)), float(np.clip(h, 0, 1))
