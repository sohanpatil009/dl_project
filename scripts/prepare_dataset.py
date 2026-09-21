"""Prepare the dataset from scratch.

This script:
1. generates the synthetic wideband dataset,
2. saves spectrograms + YOLO labels,
3. writes data.yaml,
4. generates the narrowband clips for the classifier,
5. saves a manifest of both.

Usage:
    python scripts/prepare_dataset.py --wideband-train 200 --narrowband-per-class 200
    python scripts/prepare_dataset.py --data-source radioml --image-format png
    # (radioml needs data/public/RML2016.10a_dict.pkl; see scripts/fetch_public_data.py)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, DATA_DIR, get_default_config
from src.data_loader import (
    build_and_persist_yolo_dataset, build_narrowband_classifier_dataset,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(DATA_DIR))
    p.add_argument("--wideband-train", type=int, default=None)
    p.add_argument("--wideband-val", type=int, default=None)
    p.add_argument("--wideband-test", type=int, default=None)
    p.add_argument("--narrowband-per-class", type=int, default=None)
    p.add_argument("--data-source", default="synthetic",
                   help="synthetic (built-in) | radioml (real internet data)")
    p.add_argument("--image-format", default=None,
                   help="jpg (smallest) | png (lossless) | bmp")
    p.add_argument("--radioml-pkl", default=None,
                   help="Path to RML2016.10a_dict.pkl (else data/public/ is used).")
    p.add_argument("--no-yolo", action="store_true")
    p.add_argument("--no-narrowband", action="store_true")
    args = p.parse_args()

    cfg = get_default_config()
    if args.wideband_train:
        cfg.wideband_train = args.wideband_train
    if args.wideband_val:
        cfg.wideband_val = args.wideband_val
    if args.wideband_test:
        cfg.wideband_test = args.wideband_test
    if args.narrowband_per_class:
        cfg.narrowband_per_class = args.narrowband_per_class
    cfg.data_source = args.data_source
    if args.image_format:
        cfg.image_format = args.image_format

    real_pool = None
    if cfg.data_source == "radioml":
        from src.data_loader import load_real_pool
        print("[prepare_dataset] Loading real RadioML pool ...")
        real_pool = load_real_pool(
            cfg, pkl_path=args.radioml_pkl,
            dest_dir=os.path.join(args.out, "public"))

    if not args.no_yolo:
        print(f"[prepare_dataset] Building YOLO dataset "
              f"(source={cfg.data_source}, .{cfg.image_format}) ...")
        yolo_dir = os.path.join(args.out, "yolo_dataset")
        splits = build_and_persist_yolo_dataset(
            out_dir=yolo_dir, cfg=cfg, num_classes=1, real_pool=real_pool)
        for k, v in splits.items():
            print(f"  {k}: {len(v)} images")

    if not args.no_narrowband:
        print("[prepare_dataset] Building narrowband classifier dataset ...")
        nb_dir = os.path.join(args.out, "narrowband")
        counts = build_narrowband_classifier_dataset(
            out_dir=nb_dir, cfg=cfg, real_pool=real_pool)
        for c, n in counts.items():
            print(f"  {c}: {n} clips")

    cfg.save()
    print("[prepare_dataset] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
