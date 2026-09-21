"""Signal detection stage.

Combines the spectrogram pipeline with the YOLOv5 detector to convert a
wideband complex baseband signal into a list of bounding boxes in
(time, frequency) coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import Config, get_default_config
from .spectrogram import Spectrogram, compute_spectrogram, spectrogram_to_image
from .yolo_utils import (
    Detection, load_yolov5, run_yolov5_inference,
    yolo_box_to_freq_time,
)


@dataclass
class DetectedSignal:
    """A signal region detected in a wideband spectrogram."""

    signal_id: int
    time_range: Tuple[float, float]    # seconds
    frequency_range: Tuple[float, float]  # Hz
    bounding_box_yolo: Tuple[float, float, float, float]  # (cx, cy, w, h) 0..1
    confidence: float
    class_id: int = 0
    class_name: str = "signal"
    time_axis_full: Optional[np.ndarray] = field(default=None, repr=False)
    frequency_axis_full: Optional[np.ndarray] = field(default=None, repr=False)


def detect_signals(samples: np.ndarray, fs: float, weights: str,
                   cfg: Optional[Config] = None,
                   n_fft: Optional[int] = None,
                   hop_length: Optional[int] = None,
                   conf: Optional[float] = None,
                   iou: Optional[float] = None,
                   imgsz: Optional[int] = None,
                   device: str = "") -> Tuple[Spectrogram, List[DetectedSignal]]:
    """Run the full wideband -> spectrogram -> YOLO detection pipeline.

    Parameters
    ----------
    samples : np.ndarray
        Complex baseband wideband signal.
    fs : float
        Sampling rate (Hz).
    weights : str
        Path to a trained YOLOv5 checkpoint.
    cfg : Config
        Project configuration.

    Returns
    -------
    (spectrogram, detections) tuple.
    """
    if cfg is None:
        cfg = get_default_config()
    spec = compute_spectrogram(
        samples, fs=fs,
        n_fft=n_fft or cfg.n_fft,
        hop_length=hop_length or cfg.hop_length,
        window=cfg.window,
        db_floor=cfg.db_floor,
    )
    img = spectrogram_to_image(spec, image_size=cfg.image_size)
    # Persist the image to a temp file because Ultralytics prefers paths
    import tempfile
    from PIL import Image
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    Image.fromarray(img, mode="RGB").save(tmp.name, quality=90)
    model = load_yolov5(weights)
    det_lists = run_yolov5_inference(
        model, [tmp.name], cfg=cfg, conf=conf, iou=iou, imgsz=imgsz, device=device,
    )
    detections: List[DetectedSignal] = []
    for i, det in enumerate(det_lists[0]):
        f_lo, f_hi, t_lo, t_hi = yolo_box_to_freq_time(det.cx, det.cy, det.w, det.h, spec)
        detections.append(DetectedSignal(
            signal_id=i + 1,
            time_range=(float(t_lo), float(t_hi)),
            frequency_range=(float(f_lo), float(f_hi)),
            bounding_box_yolo=(det.cx, det.cy, det.w, det.h),
            confidence=det.confidence,
            class_id=det.class_id,
            class_name=det.class_name,
            time_axis_full=spec.time_s,
            frequency_axis_full=spec.freq_hz,
        ))
    try:
        os_unlink = __import__("os").unlink
        os_unlink(tmp.name)
    except Exception:
        pass
    return spec, detections
