"""Train the YOLOv5 detector.

Usage (from project root):
    python -m src.train_detector --epochs 30 --batch 16
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from .config import Config, get_default_config, get_device, METRICS_DIR, DATA_DIR
from .yolo_utils import install_yolov5, train_yolov5


def main() -> int:
    p = argparse.ArgumentParser(description="Train YOLOv5 detector")
    p.add_argument("--data-yaml", default=str(DATA_DIR / "yolo_dataset" / "data.yaml"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--weights", default="yolov5s.pt")
    p.add_argument("--project", default="models/runs")
    p.add_argument("--name", default="detector")
    p.add_argument("--device", default="")
    args = p.parse_args()

    cfg = get_default_config()
    print(f"[train_detector] device: {get_device()}")
    install_yolov5()
    best = train_yolov5(
        data_yaml=args.data_yaml, cfg=cfg,
        epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        weights=args.weights, project=args.project, name=args.name,
        device=args.device,
    )
    out_meta = METRICS_DIR / "yolo_training.yaml"
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with open(out_meta, "w", encoding="utf-8") as f:
        yaml.safe_dump({"best_weights": str(best), "config": cfg.to_dict()}, f)
    print(f"[train_detector] best weights saved at {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
