"""Top-level data loaders.

Data sources supported (see ``cfg.data_source``):

* ``"synthetic"`` -- built-in generator following the paper's parameters.
* ``"radioml"`` -- real internet data (RadioML 2016.10A clips mixed into
  wideband captures; see ``src/public_data.py``).
* existing on-disk ``yolo_dataset/`` -- picked up via
  :func:`discover_external_dataset`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Config, get_default_config
from .preprocessing import (
    WidebandSample, generate_wideband, generate_narrowband, bandpass_extract,
)
from .dataset_utils import (
    save_wideband_dataset, split_yolo_dataset, write_yolo_yaml,
    save_narrowband_clips, StoredSample,
)


# ---------------------------------------------------------------------------
# Synthetic wideband generation
# ---------------------------------------------------------------------------
def build_synthetic_wideband(cfg: Optional[Config] = None,
                             split: str = "train") -> List[WidebandSample]:
    """Build a synthetic wideband dataset following the paper's recipe.

    Parameters
    ----------
    cfg : Config
        Project configuration.
    split : str
        One of {"train", "val", "test"}.
    """
    if cfg is None:
        cfg = get_default_config()
    if split == "train":
        n = cfg.wideband_train
        snr = cfg.snr_train_db
        rng = np.random.default_rng(cfg.seed)
    elif split == "val":
        n = cfg.wideband_val
        snr = cfg.snr_train_db
        rng = np.random.default_rng(cfg.seed + 1)
    elif split == "test":
        n = cfg.wideband_test
        # At test time we draw samples at the *train* SNR to compare
        # with the paper; SNR sweeps are done separately.
        snr = cfg.snr_train_db
        rng = np.random.default_rng(cfg.seed + 2)
    else:
        raise ValueError(f"Unknown split: {split}")
    num_samples = int(cfg.fs * cfg.duration)
    samples: List[WidebandSample] = []
    for _ in range(n):
        s = generate_wideband(
            num_samples=num_samples,
            fs=cfg.fs,
            snr_db=snr,
            mod_classes=cfg.mod_classes,
            min_signals=cfg.min_signals,
            max_signals=cfg.max_signals,
            bandwidth=cfg.bandwidth,
            padding=cfg.padding,
            rng=rng,
        )
        samples.append(s)
    return samples


def build_snr_sweep(cfg: Optional[Config] = None,
                    per_snr: int = 50) -> Dict[float, List[WidebandSample]]:
    """Generate one wideband set per SNR value for the evaluation table."""
    if cfg is None:
        cfg = get_default_config()
    out: Dict[float, List[WidebandSample]] = {}
    num_samples = int(cfg.fs * cfg.duration)
    for i, snr in enumerate(cfg.snr_eval_db):
        rng = np.random.default_rng(cfg.seed + 1000 + i)
        out[snr] = [
            generate_wideband(
                num_samples=num_samples,
                fs=cfg.fs,
                snr_db=snr,
                mod_classes=cfg.mod_classes,
                min_signals=cfg.min_signals,
                max_signals=cfg.max_signals,
                bandwidth=cfg.bandwidth,
                padding=cfg.padding,
                rng=rng,
            )
            for _ in range(per_snr)
        ]
    return out


# ---------------------------------------------------------------------------
# End-to-end: build + persist a YOLO dataset
# ---------------------------------------------------------------------------
def _wideband_for_split(cfg: Config, split: str,
                        real_pool: Optional[Dict[str, List[np.ndarray]]] = None
                        ) -> List[WidebandSample]:
    """Return wideband samples for one split from synthetic or real data."""
    if cfg.data_source == "radioml":
        if real_pool is None:
            raise ValueError("data_source='radioml' requires a real_pool; "
                             "see src/public_data.radioml_to_clips.")
        from .public_data import generate_wideband_from_real
        n_map = {"train": cfg.wideband_train, "val": cfg.wideband_val,
                 "test": cfg.wideband_test}
        seed_map = {"train": cfg.seed, "val": cfg.seed + 1, "test": cfg.seed + 2}
        if split not in n_map:
            raise ValueError(f"Unknown split: {split}")
        rng = np.random.default_rng(seed_map[split])
        num_samples = int(cfg.fs * cfg.duration)
        return [
            generate_wideband_from_real(
                num_samples=num_samples, fs=cfg.fs, snr_db=cfg.snr_train_db,
                real_pool=real_pool, mod_classes=cfg.mod_classes,
                min_signals=cfg.min_signals, max_signals=cfg.max_signals,
                bandwidth=cfg.bandwidth, padding=cfg.padding, rng=rng)
            for _ in range(n_map[split])
        ]
    return build_synthetic_wideband(cfg, split=split)


def build_and_persist_yolo_dataset(out_dir: str, cfg: Optional[Config] = None,
                                   yolo_class_map: Optional[Dict[str, int]] = None,
                                   num_classes: int = 1,
                                   image_format: Optional[str] = None,
                                   real_pool: Optional[Dict[str, List[np.ndarray]]] = None
                                   ) -> Dict[str, List[str]]:
    """Generate train/val/test, save spectrogram images + labels, write YAML."""
    if cfg is None:
        cfg = get_default_config()
    image_format = image_format or cfg.image_format
    raw_dir = Path(out_dir) / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_samples: List[WidebandSample] = []
    splits_meta: Dict[str, List[WidebandSample]] = {}
    for split in ("train", "val", "test"):
        samps = _wideband_for_split(cfg, split=split, real_pool=real_pool)
        splits_meta[split] = samps
        all_samples.extend(samps)
    # Save as one flat folder
    stored = save_wideband_dataset(
        samples=all_samples,
        out_dir=str(raw_dir),
        spec_image_size=cfg.image_size,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        window=cfg.window,
        db_floor=cfg.db_floor,
        make_yolo_labels=True,
        yolo_class_map=yolo_class_map,
        bandwidth_hz=cfg.bandwidth,
        padding_hz=cfg.padding,
        num_classes=num_classes,
        image_format=image_format,
    )
    # Now reorganise into train/val/test based on the original split.
    # We re-split deterministically to recover which sample belongs to which
    # split (the `stored` list has the same ordering as `all_samples`).
    counts = {sp: len(splits_meta[sp]) for sp in ("train", "val", "test")}
    edges = np.cumsum([0, counts["train"], counts["val"], counts["test"]])
    # Re-shuffle with the same seed for reproducibility
    rng = np.random.default_rng(cfg.seed + 999)
    perm = np.arange(len(all_samples))
    rng.shuffle(perm)
    # We must keep the per-split assignment consistent with the saved files.
    # Strategy: re-do the split inside `split_yolo_dataset` using the *same*
    # ratios and seed by writing custom per-split files instead of relying
    # on `split_yolo_dataset`. This is simpler and deterministic.
    img_root = Path(out_dir) / "images"
    lbl_root = Path(out_dir) / "labels"
    for sub in ("train", "val", "test"):
        (img_root / sub).mkdir(parents=True, exist_ok=True)
        (lbl_root / sub).mkdir(parents=True, exist_ok=True)
    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for s in stored:
        # Decide which split using the original indices
        if s.sample_id < edges[1]:
            sub = "train"
        elif s.sample_id < edges[2]:
            sub = "val"
        else:
            sub = "test"
        import shutil
        src_img = Path(s.spec_path)
        src_lbl = Path(s.label_path) if s.label_path else None
        dst_img = img_root / sub / src_img.name
        dst_lbl = lbl_root / sub / src_lbl.name if src_lbl else None
        shutil.move(str(src_img), dst_img)
        if src_lbl and src_lbl.exists():
            shutil.move(str(src_lbl), dst_lbl)
        splits[sub].append(str(dst_img))
    # Cleanup the _raw folder
    try:
        (raw_dir / "images").rmdir()
    except OSError:
        pass
    try:
        (raw_dir / "labels").rmdir()
    except OSError:
        pass
    try:
        (raw_dir / "index.json").unlink()
    except OSError:
        pass
    try:
        raw_dir.rmdir()
    except OSError:
        pass
    # Write data.yaml
    write_yolo_yaml(
        out_path=str(Path(out_dir) / "data.yaml"),
        dataset_dir=out_dir,
        num_classes=num_classes,
        class_names=(["signal"] if num_classes == 1
                     else [f"class_{i}" for i in range(num_classes)]),
    )
    return splits


# ---------------------------------------------------------------------------
# Narrowband dataset for the classifier
# ---------------------------------------------------------------------------
def build_narrowband_classifier_dataset(out_dir: str, cfg: Optional[Config] = None,
                                        per_class: Optional[int] = None,
                                        real_pool: Optional[Dict[str, List[np.ndarray]]] = None
                                        ) -> Dict[str, int]:
    """Generate IQ clips for the modulation classifier (per-class folders).

    When ``cfg.data_source == "radioml"`` (or `real_pool` is given), real
    RadioML clips back the overlapping classes; the rest (NOISE/NO_SIGNAL
    and any pool misses) fall back to local synthesis.
    """
    if cfg is None:
        cfg = get_default_config()
    per_class = per_class or cfg.narrowband_per_class
    num_samples = cfg.clf_seq_len
    rng = np.random.default_rng(cfg.seed + 4242)
    use_real = (cfg.data_source == "radioml") and (real_pool is not None)
    by_class: Dict[str, List[np.ndarray]] = {}
    for cls in cfg.mod_classes:
        if cls == "NO_SIGNAL":
            # Empty clip = zeros
            by_class[cls] = [np.zeros(num_samples, dtype=np.complex128) for _ in range(per_class)]
            continue
        if use_real and real_pool is not None and cls in real_pool and real_pool[cls]:
            pool = real_pool[cls]
            idx = rng.choice(len(pool), size=per_class, replace=len(pool) < per_class)
            by_class[cls] = [np.asarray(pool[int(i)][:num_samples]).astype(np.complex128)
                             for i in idx]
            continue
        clips: List[np.ndarray] = []
        for _ in range(per_class):
            sig = generate_narrowband(
                num_samples=num_samples,
                modulation=cls,
                snr_db=cfg.snr_train_db,
                rng=rng,
                beta=cfg.roll_off,
            )
            clips.append(sig)
        by_class[cls] = clips
    save_narrowband_clips(by_class, out_dir=out_dir, sample_len=num_samples)
    return {c: len(v) for c, v in by_class.items()}


def load_real_pool(cfg: Optional[Config] = None,
                   pkl_path: Optional[str] = None,
                   dest_dir: Optional[str] = None) -> Dict[str, List[np.ndarray]]:
    """Download (if needed) + load RadioML and return a tiled-clip pool.

    On Kaggle with an attached dataset, pass ``pkl_path`` pointing into
    ``/kaggle/input/...`` to skip the download entirely.
    """
    from .public_data import (download_radioml, find_kaggle_radioml,
                              load_radioml_pkl, radioml_to_clips)
    if cfg is None:
        cfg = get_default_config()
    if pkl_path is None:
        pkl_path = find_kaggle_radioml()
    if pkl_path is None:
        pkl_path = download_radioml(dest_dir or str(Path(cfg.data_dir) / "public"),
                                    url=cfg.radioml_url)
    data = load_radioml_pkl(pkl_path)
    rng = np.random.default_rng(cfg.seed + 777)
    # Generous pool so wideband tiling + classifier splits never run dry
    pool = radioml_to_clips(data, snr_db=cfg.radioml_snr_db,
                            per_class=max(1000, cfg.narrowband_per_class * 2),
                            rng=rng, include_extra=cfg.radioml_include_extra)
    return pool


# ---------------------------------------------------------------------------
# Public dataset discovery (for external data)
# ---------------------------------------------------------------------------
def discover_external_dataset(root: str) -> Optional[Dict[str, str]]:
    """Inspect a directory and try to find a dataset that matches our format.

    The expected layout is::

        root/
        ├── images/{train,val,test}/*.{jpg,png,bmp}
        ├── labels/{train,val,test}/*.txt
        └── data.yaml

    Returns ``None`` if the layout does not match.
    """
    root = Path(root)
    if not (root / "images").is_dir():
        return None
    if not (root / "labels").is_dir():
        return None
    if not (root / "data.yaml").is_file():
        return None
    return {
        "images": str(root / "images"),
        "labels": str(root / "labels"),
        "data_yaml": str(root / "data.yaml"),
    }
