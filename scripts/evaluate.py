"""Evaluation entry point.

Usage:
    python scripts/evaluate.py --detector-weights models/runs/detector/weights/best.pt \
                               --classifier-ckpt models/classifier.pt \
                               --out results/metrics
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.config import get_default_config, get_device, METRICS_DIR
from src.evaluate import (
    evaluate_detector, evaluate_classifier,
    snr_sweep_detector, snr_sweep_classifier,
)
from src.data_loader import build_synthetic_wideband
from src.classifier import build_model, predict_classifier
from src.yolo_utils import load_yolov5
from src.preprocessing import generate_narrowband
from src.spectrogram import compute_spectrogram


def ground_truth_boxes(samples, cfg):
    boxes_per_image = []
    for s in samples:
        spec = compute_spectrogram(s.samples, fs=s.fs, n_fft=cfg.n_fft,
                                   hop_length=cfg.hop_length, window=cfg.window,
                                   db_floor=cfg.db_floor)
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
        boxes_per_image.append(boxes)
    return boxes_per_image


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--detector-weights", required=True)
    p.add_argument("--classifier-ckpt", required=True)
    p.add_argument("--out", default=str(METRICS_DIR))
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--snr-sweep", action="store_true")
    args = p.parse_args()

    cfg = get_default_config()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Detector evaluation ----------
    print("[evaluate] Loading detector ...")
    detector = load_yolov5(args.detector_weights)
    test_samples = build_synthetic_wideband(cfg, split="test")
    gts = ground_truth_boxes(test_samples, cfg)
    det_metrics = evaluate_detector(detector, list(zip(test_samples, gts)), cfg=cfg)
    print(f"[evaluate] Detector P/R/F1/IoU = "
          f"{det_metrics['precision']:.3f}/{det_metrics['recall']:.3f}/"
          f"{det_metrics['f1']:.3f}/{det_metrics['mean_iou']:.3f}")
    with open(out_dir / "detector_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in det_metrics.items() if k != "per_image"}, f, indent=2)

    # ---------- Classifier evaluation ----------
    print("[evaluate] Loading classifier ...")
    device = get_device()
    ckpt = torch.load(args.classifier_ckpt, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    classes = ckpt["classes"]
    model = build_model(arch=arch, num_classes=len(classes), in_channels=2,
                        seq_len=cfg.clf_seq_len, use_transfer=False)
    model.load_state_dict(ckpt["state_dict"])
    # Build a held-out test set at training SNR
    rng = np.random.default_rng(cfg.seed + 5)
    Xs, ys = [], []
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    for cls in classes:
        if cls == "NO_SIGNAL":
            for _ in range(args.n_test):
                Xs.append(np.zeros((2, cfg.clf_seq_len), dtype=np.float32))
                ys.append(cls_to_idx[cls])
            continue
        for _ in range(args.n_test):
            clip = generate_narrowband(num_samples=cfg.clf_seq_len,
                                       modulation=cls, snr_db=cfg.snr_train_db,
                                       rng=rng, beta=cfg.roll_off)
            p = np.mean(np.abs(clip) ** 2)
            if p > 0:
                clip = clip / np.sqrt(p)
            Xs.append(np.stack([clip.real, clip.imag], axis=0).astype(np.float32))
            ys.append(cls_to_idx[cls])
    X = np.stack(Xs, axis=0)
    y = np.array(ys, dtype=np.int64)
    preds, probs = predict_classifier(model, X, device=device)
    cls_metrics = evaluate_classifier(y, preds, classes, out_dir=str(out_dir))
    print(f"[evaluate] Classifier accuracy = {cls_metrics['accuracy']:.3f}")

    # ---------- SNR sweeps (optional) ----------
    if args.snr_sweep:
        print("[evaluate] Running SNR sweep (detector) ...")
        det_snr = snr_sweep_detector(detector, cfg=cfg, per_snr=20)
        with open(out_dir / "detector_snr_sweep.json", "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in det_snr.items()}, f, indent=2)
        print("[evaluate] Running SNR sweep (classifier) ...")
        clf_snr = snr_sweep_classifier(model, classes, cfg=cfg, per_snr_per_class=40)
        with open(out_dir / "classifier_snr_sweep.json", "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in clf_snr.items()}, f, indent=2)
    print("[evaluate] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
