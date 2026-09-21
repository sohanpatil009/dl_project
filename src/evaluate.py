"""Evaluation utilities.

Implements:
* Detection metrics: precision, recall, mAP@0.5, IoU per detection
* Classification metrics: accuracy, precision, recall, F1, confusion matrix
* SNR-sweep evaluation for both stages
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
)

from .config import Config, get_default_config, get_device
from .preprocessing import generate_wideband
from .spectrogram import compute_spectrogram, spectrogram_to_image
from .yolo_utils import (
    Detection, load_yolov5, run_yolov5_inference,
)
from .signal_detection import detect_signals
from .signal_extraction import extract_narrowband
from .classifier import build_model, predict_classifier
from .visualization import plot_confusion_matrix


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------
def iou_xywh(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]) -> float:
    """IoU between two YOLO boxes (cx, cy, w, h) in normalised 0..1 units."""
    ax1 = a[0] - a[2] / 2
    ay1 = a[1] - a[3] / 2
    ax2 = a[0] + a[2] / 2
    ay2 = a[1] + a[3] / 2
    bx1 = b[0] - b[2] / 2
    by1 = b[1] - b[3] / 2
    bx2 = b[0] + b[2] / 2
    by2 = b[1] + b[3] / 2
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / (union + 1e-12))


def match_detections(preds: List[Tuple[float, float, float, float]],
                     gts: List[Tuple[float, float, float, float]],
                     iou_threshold: float = 0.5) -> Tuple[int, int, int, List[float]]:
    """Greedy IoU-based matching.

    Returns (tp, fp, fn, iou_list_for_tp).
    """
    matched_gt = set()
    tp, fp = 0, 0
    iou_list: List[float] = []
    for p in preds:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in matched_gt:
                continue
            i = iou_xywh(p, g)
            if i > best_iou:
                best_iou, best_j = i, j
        if best_iou >= iou_threshold and best_j >= 0:
            tp += 1
            matched_gt.add(best_j)
            iou_list.append(best_iou)
        else:
            fp += 1
    fn = len(gts) - len(matched_gt)
    return tp, fp, fn, iou_list


def evaluate_detector(detector, samples_signals, cfg: Optional[Config] = None,
                      conf: Optional[float] = None,
                      iou_thr: float = 0.3,
                      device: Optional[torch.device] = None
                      ) -> Dict[str, float]:
    """Evaluate the YOLOv5 detector against ground-truth boxes.

    `samples_signals` is a list of (WidebandSample, list_of_yolo_gt_boxes).
    """
    if cfg is None:
        cfg = get_default_config()
    if device is None:
        device = get_device()
    conf = conf if conf is not None else cfg.yolo_conf
    tp_total = fp_total = fn_total = 0
    iou_all: List[float] = []
    per_image: List[Dict] = []
    from PIL import Image
    import tempfile
    tmp_files: List[str] = []
    image_paths: List[str] = []
    gts_per_image: List[List[Tuple[float, float, float, float]]] = []
    for s, gt_boxes in samples_signals:
        spec = compute_spectrogram(s.samples, fs=s.fs, n_fft=cfg.n_fft,
                                   hop_length=cfg.hop_length, window=cfg.window,
                                   db_floor=cfg.db_floor)
        img = spectrogram_to_image(spec, image_size=cfg.image_size)
        t = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        Image.fromarray(img, mode="RGB").save(t.name, quality=90)
        image_paths.append(t.name)
        tmp_files.append(t.name)
        gts_per_image.append(gt_boxes)
    det_lists = run_yolov5_inference(
        detector, image_paths, cfg=cfg, conf=conf, iou=cfg.yolo_iou,
        imgsz=cfg.yolo_input_size, device=str(device),
    )
    for i, dets in enumerate(det_lists):
        preds = [(d.cx, d.cy, d.w, d.h) for d in dets]
        gts = gts_per_image[i]
        tp, fp, fn, iou_list = match_detections(preds, gts, iou_threshold=iou_thr)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        iou_all.extend(iou_list)
        per_image.append({
            "image": image_paths[i],
            "n_pred": len(preds), "n_gt": len(gts),
            "tp": tp, "fp": fp, "fn": fn,
            "mean_iou": float(np.mean(iou_list)) if iou_list else 0.0,
        })
    # Clean up tmp files
    for p in tmp_files:
        try:
            os.unlink(p)
        except OSError:
            pass
    precision = tp_total / (tp_total + fp_total + 1e-12)
    recall = tp_total / (tp_total + fn_total + 1e-12)
    mean_iou = float(np.mean(iou_all)) if iou_all else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + 1e-12),
        "mean_iou": mean_iou,
        "tp": tp_total, "fp": fp_total, "fn": fn_total,
        "n_images": len(det_lists),
        "per_image": per_image,
    }


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------
def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray,
                        class_names: Sequence[str],
                        out_dir: Optional[str] = None) -> Dict:
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    metrics = {
        "accuracy": acc,
        "precision_per_class": prec.tolist(),
        "recall_per_class": rec.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist(),
        "precision_macro": float(np.mean(prec)),
        "recall_macro": float(np.mean(rec)),
        "f1_macro": float(np.mean(f1)),
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(cm, class_names, normalize=True,
                              out_path=os.path.join(out_dir, "confusion_matrix.png"))
        with open(os.path.join(out_dir, "classification_metrics.json"), "w",
                  encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    return metrics


# ---------------------------------------------------------------------------
# SNR sweeps
# ---------------------------------------------------------------------------
def snr_sweep_detector(detector, cfg: Optional[Config] = None,
                       per_snr: int = 50,
                       iou_thr: float = 0.3) -> Dict[float, Dict[str, float]]:
    if cfg is None:
        cfg = get_default_config()
    from .data_loader import build_snr_sweep
    sweep = build_snr_sweep(cfg, per_snr=per_snr)
    out: Dict[float, Dict[str, float]] = {}
    for snr, samples in sweep.items():
        # ground truth boxes (cx, cy, w, h)
        gts = []
        for s in samples:
            spec = compute_spectrogram(s.samples, fs=s.fs, n_fft=cfg.n_fft,
                                       hop_length=cfg.hop_length, window=cfg.window,
                                       db_floor=cfg.db_floor)
            n_freq, n_time = spec.shape
            f_lo, f_hi = float(spec.freq_hz.min()), float(spec.freq_hz.max())
            t_lo, t_hi = float(spec.time_s.min()), float(spec.time_s.max())
            boxes = []
            for nbs in s.signals:
                cx = (nbs.carrier_offset_hz - f_lo) / (f_hi - f_lo + 1e-12)
                cy = (0.5 * (t_lo + t_hi) - t_lo) / (t_hi - t_lo + 1e-12)
                w = (cfg.bandwidth + 2 * cfg.padding) / (f_hi - f_lo + 1e-12)
                h = 1.0
                boxes.append((float(np.clip(cx, 0, 1)), float(np.clip(cy, 0, 1)),
                              float(np.clip(w, 0, 1)), float(np.clip(h, 0, 1))))
            gts.append(boxes)
        pairs = list(zip(samples, gts))
        m = evaluate_detector(detector, pairs, cfg=cfg, iou_thr=iou_thr)
        out[snr] = {k: v for k, v in m.items() if k != "per_image"}
    return out


def snr_sweep_classifier(classifier_model, class_names, cfg: Optional[Config] = None,
                         per_snr_per_class: int = 50) -> Dict[float, Dict]:
    if cfg is None:
        cfg = get_default_config()
    device = get_device()
    out: Dict[float, Dict] = {}
    classes = [c for c in class_names if c not in ("NO_SIGNAL",)]
    for snr in cfg.snr_eval_db:
        # Offset keeps the seed non-negative for negative-SNR bins
        rng = np.random.default_rng(int((snr + 100) * 1000) + 7)
        Xs, ys = [], []
        for ci, cls in enumerate(classes):
            for _ in range(per_snr_per_class):
                from .preprocessing import generate_narrowband
                clip = generate_narrowband(num_samples=cfg.clf_seq_len,
                                           modulation=cls, snr_db=snr, rng=rng,
                                           beta=cfg.roll_off)
                p = np.mean(np.abs(clip) ** 2)
                if p > 0:
                    clip = clip / np.sqrt(p)
                Xs.append(np.stack([clip.real, clip.imag], axis=0).astype(np.float32))
                ys.append(ci)
        X = np.stack(Xs, axis=0)
        y = np.array(ys, dtype=np.int64)
        preds, _ = predict_classifier(classifier_model, X, device=device)
        m = evaluate_classifier(y, preds, classes)
        out[snr] = m
    return out
