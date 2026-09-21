"""Train the modulation classifier.

Usage:
    python -m src.train_classifier --data data/narrowband --arch resnet18
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .config import Config, get_default_config, get_device, MODELS_DIR, METRICS_DIR
from .classifier import build_model, train_classifier, TrainConfig, predict_classifier


def load_narrowband_manifest(data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load IQ clips saved by `data_loader.build_narrowband_classifier_dataset`."""
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found in {data_dir}. "
            "Run `python -m src.data_loader` (or call "
            "`build_narrowband_classifier_dataset`) first."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest: List[List[str]] = json.load(f)
    class_names = sorted({m[0] for m in manifest})
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    X: List[np.ndarray] = []
    y: List[int] = []
    for cls, path in manifest:
        arr = np.load(path)
        # arr is complex; convert to (2, L)
        arr = np.stack([arr.real, arr.imag], axis=0).astype(np.float32)
        X.append(arr)
        y.append(cls_to_idx[cls])
    X_arr = np.stack(X, axis=0)
    y_arr = np.array(y, dtype=np.int64)
    return X_arr, y_arr, class_names


def stratified_split(y: np.ndarray, train: float = 0.7, val: float = 0.15,
                     seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_train = max(1, int(len(idx) * train))
        n_val = max(1, int(len(idx) * val))
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def main() -> int:
    p = argparse.ArgumentParser(description="Train modulation classifier")
    p.add_argument("--data", default=str(MODELS_DIR.parent / "data" / "narrowband"))
    p.add_argument("--arch", default=None, help="cnn1d|resnet18|resnet34")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--no-transfer", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--out", default=str(MODELS_DIR / "classifier.pt"))
    p.add_argument("--history", default=str(METRICS_DIR / "clf_history.json"))
    args = p.parse_args()

    cfg = get_default_config()
    arch = args.arch or cfg.clf_arch
    X, y, classes = load_narrowband_manifest(args.data)
    print(f"[train_classifier] dataset: {X.shape}  classes={classes}")
    tr, va, te = stratified_split(y, train=0.7, val=0.15, seed=cfg.seed)
    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]
    X_te, y_te = X[te], y[te]
    print(f"[train_classifier] train={len(tr)}  val={len(va)}  test={len(te)}")
    model = build_model(
        arch=arch, num_classes=len(classes), in_channels=2,
        seq_len=cfg.clf_seq_len, use_transfer=(cfg.clf_use_transfer and not args.no_transfer),
    )
    tcfg = TrainConfig(
        epochs=args.epochs or cfg.clf_epochs,
        batch_size=args.batch or cfg.clf_batch_size,
        lr=args.lr or cfg.clf_lr,
        weight_decay=cfg.clf_weight_decay,
        patience=cfg.clf_patience,
        use_augmentation=(cfg.clf_use_augmentation and not args.no_augment),
        device=str(get_device()),
    )
    history = train_classifier(model, X_tr, y_tr, X_va, y_va,
                               cfg=tcfg, num_classes=len(classes), verbose=True)
    # Final test
    preds, probs = predict_classifier(model, X_te, device=get_device())
    test_acc = float((preds == y_te).mean())
    print(f"[train_classifier] test acc = {test_acc:.4f}")
    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save({
        "state_dict": model.state_dict(),
        "arch": arch,
        "classes": classes,
        "test_acc": test_acc,
        "config": cfg.to_dict(),
    }, args.out)
    Path(args.history).parent.mkdir(parents=True, exist_ok=True)
    with open(args.history, "w", encoding="utf-8") as f:
        json.dump({"history": history, "test_acc": test_acc, "classes": classes}, f, indent=2)
    print(f"[train_classifier] saved model to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
