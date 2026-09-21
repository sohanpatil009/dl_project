"""Fetch the public internet dataset (RadioML 2016.10A) and build our sets.

This is the "use our own (internet) data" entry point:

* Colab / local: downloads the Zenodo mirror (~213 MB) into data/public/.
* Kaggle: pass --radioml-pkl /kaggle/input/<dataset>/RML2016.10a_dict.pkl
  (attached via + Add Data) to skip the download.

Examples:
    python scripts/fetch_public_data.py --out data --narrowband-per-class 400
    python scripts/fetch_public_data.py --out /kaggle/working/wsr \\
        --radioml-pkl /kaggle/input/rml2016-10a/RML2016.10a_dict.pkl \\
        --image-format png --wideband-train 300

What it builds (both types):
    <out>/narrowband/           # real 2x128 clips per class (.npy + manifest.json)
    <out>/yolo_dataset/         # real-mix spectrogram images (.jpg/.png/.bmp)
                                #   + YOLO .txt labels + data.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_default_config
from src.data_loader import (
    build_and_persist_yolo_dataset,
    build_narrowband_classifier_dataset,
    load_real_pool,
)
from src.dataset_utils import save_narrowband_clips
from src.public_data import radioml_summary, load_radioml_pkl


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data")
    p.add_argument("--radioml-pkl", default=None,
                   help="Path to an existing RML2016.10a_dict.pkl (e.g. Kaggle input).")
    p.add_argument("--public-dir", default=None,
                   help="Download/extract dir (default: <out>/public).")
    p.add_argument("--wideband-train", type=int, default=None)
    p.add_argument("--wideband-val", type=int, default=None)
    p.add_argument("--wideband-test", type=int, default=None)
    p.add_argument("--narrowband-per-class", type=int, default=None)
    p.add_argument("--image-format", default=None,
                   help="jpg (smallest) | png (lossless) | bmp")
    p.add_argument("--include-extra", action="store_true",
                   help="Also keep non-paper RadioML classes (CPFSK/GFSK/PAM4/AM-*/WBFM).")
    p.add_argument("--no-yolo", action="store_true")
    p.add_argument("--no-narrowband", action="store_true")
    args = p.parse_args()

    cfg = get_default_config()
    cfg.data_source = "radioml"
    if args.wideband_train:
        cfg.wideband_train = args.wideband_train
    if args.wideband_val:
        cfg.wideband_val = args.wideband_val
    if args.wideband_test:
        cfg.wideband_test = args.wideband_test
    if args.narrowband_per_class:
        cfg.narrowband_per_class = args.narrowband_per_class
    if args.image_format:
        cfg.image_format = args.image_format
    if args.include_extra:
        cfg.radioml_include_extra = True

    import numpy as np
    print("[fetch] loading RadioML 2016.10A ...")
    pool = load_real_pool(cfg, pkl_path=args.radioml_pkl,
                          dest_dir=args.public_dir or str(Path(args.out) / "public"))
    print(f"[fetch] pool classes: {sorted(pool)} "
          f"({sum(len(v) for v in pool.values())} clips)")
    print("[fetch] note: BPSK/QPSK/8PSK/16QAM/64QAM overlap the paper; "
          "NOISE/NO_SIGNAL are synthesised locally.")

    if not args.no_narrowband:
        print("[fetch] building narrowband set from REAL clips ...")
        counts = build_narrowband_classifier_dataset(
            out_dir=str(Path(args.out) / "narrowband"), cfg=cfg, real_pool=pool)
        for c, n in sorted(counts.items()):
            print(f"  {c}: {n} clips")

    if not args.no_yolo:
        print(f"[fetch] building YOLO set from REAL mixes (.{cfg.image_format}) ...")
        splits = build_and_persist_yolo_dataset(
            out_dir=str(Path(args.out) / "yolo_dataset"), cfg=cfg,
            num_classes=1, real_pool=pool)
        for k, v in splits.items():
            print(f"  {k}: {len(v)} images")

    cfg.save(str(Path(args.out) / "public_config.yaml"))
    print("[fetch] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
