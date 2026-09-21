"""Run end-to-end inference on a saved dataset sample.

Usage:
    python scripts/inference.py --detector-weights models/runs/detector/weights/best.pt \
                               --classifier-ckpt models/classifier.pt \
                               --num-samples 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import get_default_config, PREDICTIONS_DIR
from src.data_loader import build_synthetic_wideband
from src.inference import run_end_to_end_inference, results_to_dict


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--detector-weights", required=True)
    p.add_argument("--classifier-ckpt", required=True)
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--out", default=str(PREDICTIONS_DIR / "end_to_end.json"))
    args = p.parse_args()

    cfg = get_default_config()
    samples = build_synthetic_wideband(cfg, split="test")
    out = []
    for i, s in enumerate(samples[: args.num_samples]):
        print(f"[inference] Running on test sample {i} ...")
        _, dets, ext, res = run_end_to_end_inference(
            samples=s.samples, fs=s.fs,
            detector_weights=args.detector_weights,
            classifier_ckpt=args.classifier_ckpt, cfg=cfg,
        )
        out.append({
            "sample_id": i,
            "n_signals_detected": len(dets),
            "results": results_to_dict(res),
        })
        for r in res:
            print(f"  signal {r.signal_id}: f={r.frequency_range_hz} "
                  f"t={r.time_range_s}  mod={r.predicted_modulation} "
                  f"det={r.detector_confidence:.2f} "
                  f"clf={r.classifier_confidence:.2f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[inference] Saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
