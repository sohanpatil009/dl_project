"""End-to-end inference.

The function `run_end_to_end_inference` ties together every component:

    wideband I/Q -> spectrogram -> YOLO detection -> narrowband extraction
                  -> CNN/ResNet classification -> structured result

It is intentionally simple to use so it can be imported from a notebook
or a Flask service.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import Config, get_default_config, get_device
from .signal_detection import DetectedSignal, detect_signals
from .signal_extraction import ExtractedSignal, extract_narrowband
from .classifier import build_model, predict_classifier


@dataclass
class EndToEndResult:
    """One result per detected signal."""

    signal_id: int
    time_range_s: Tuple[float, float]
    frequency_range_hz: Tuple[float, float]
    detector_confidence: float
    predicted_modulation: str
    classifier_confidence: float
    classifier_topk: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class EndToEndReport:
    """Top-level output for one wideband capture."""

    input_fs: float
    n_samples: int
    signals: List[EndToEndResult]


def run_end_to_end_inference(samples: np.ndarray, fs: float,
                             detector_weights: str, classifier_ckpt: str,
                             cfg: Optional[Config] = None,
                             conf: Optional[float] = None,
                             iou: Optional[float] = None,
                             imgsz: Optional[int] = None,
                             topk: int = 3,
                             device: Optional[torch.device] = None
                             ) -> Tuple[Any, List[DetectedSignal], List[ExtractedSignal], List[EndToEndResult]]:
    """End-to-end inference on one wideband complex baseband signal.

    Returns
    -------
    (spectrogram, detections, extracted_signals, results)
    """
    if cfg is None:
        cfg = get_default_config()
    if device is None:
        device = get_device()
    spec, detections = detect_signals(
        samples, fs=fs, weights=detector_weights, cfg=cfg,
        conf=conf, iou=iou, imgsz=imgsz, device=str(device),
    )
    extracted = extract_narrowband(samples, fs, detections,
                                     target_len=cfg.clf_seq_len, cfg=cfg)
    # Load classifier
    ckpt = torch.load(classifier_ckpt, map_location=device, weights_only=False)
    arch = ckpt.get("arch", cfg.clf_arch)
    classes = ckpt.get("classes", list(cfg.mod_classes))
    model = build_model(arch=arch, num_classes=len(classes), in_channels=2,
                        seq_len=cfg.clf_seq_len, use_transfer=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    # Classify
    results: List[EndToEndResult] = []
    if extracted:
        X = np.stack([e.classifier_input for e in extracted], axis=0)
        preds, probs = predict_classifier(model, X, device=device)
        for k, ex in enumerate(extracted):
            topk_idx = np.argsort(-probs[k])[:topk]
            topk_list = [(classes[i], float(probs[k, i])) for i in topk_idx]
            results.append(EndToEndResult(
                signal_id=ex.signal_id,
                time_range_s=ex.time_range,
                frequency_range_hz=ex.frequency_range,
                detector_confidence=detections[k].confidence,
                predicted_modulation=classes[int(preds[k])],
                classifier_confidence=float(probs[k, preds[k]]),
                classifier_topk=topk_list,
            ))
    return spec, detections, extracted, results


def results_to_dict(report: List[EndToEndResult]) -> List[Dict[str, Any]]:
    """Convert a list of EndToEndResult to JSON-serialisable dicts."""
    out: List[Dict[str, Any]] = []
    for r in report:
        out.append({
            "signal_id": r.signal_id,
            "frequency_range": list(r.frequency_range_hz),
            "time_range": list(r.time_range_s),
            "detector_confidence": r.detector_confidence,
            "predicted_modulation": r.predicted_modulation,
            "classifier_confidence": r.classifier_confidence,
            "topk": r.classifier_topk,
        })
    return out
