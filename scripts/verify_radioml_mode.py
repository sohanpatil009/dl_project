"""Verify radioml-mode disk plumbing with a mock pool (no download)."""
import sys, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.config import get_default_config
from src.public_data import radioml_to_clips
from src.data_loader import build_and_persist_yolo_dataset, build_narrowband_classifier_dataset

# Mock RadioML dict: 5 overlap mods x SNR 18 x 12 clips
rng = np.random.default_rng(0)
mock = {}
for mod in ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64"]:
    mock[(mod, 18)] = rng.normal(0, 1, (12, 2, 128)).astype(np.float32)

pool = radioml_to_clips(mock, snr_db=18.0, per_class=12, rng=np.random.default_rng(1))
print("pool:", {k: len(v) for k, v in pool.items()})

cfg = get_default_config()
cfg.data_source = "radioml"
cfg.image_format = "png"
cfg.wideband_train = 4
cfg.wideband_val = 2
cfg.wideband_test = 2
cfg.narrowband_per_class = 6

out = Path("data_verify_radioml")
if out.exists():
    shutil.rmtree(out)
splits = build_and_persist_yolo_dataset(str(out / "yolo_dataset"), cfg=cfg,
                                        num_classes=1, real_pool=pool)
print("yolo splits:", {k: len(v) for k, v in splits.items()})
print("sample ext:", Path(splits["train"][0]).suffix)
assert Path(splits["train"][0]).suffix == ".png"
counts = build_narrowband_classifier_dataset(str(out / "narrowband"), cfg=cfg,
                                             real_pool=pool)
print("narrowband:", counts)
assert counts.get("BPSK", 0) == 6
shutil.rmtree(out)
print("RADIOML-MODE DISK CHECK OK")
