"""Simulate the local-Jupyter notebook cells (§6-11, §19-21, §24) on CPU.

Tiny sizes so it finishes in ~1 min. Saves figures exactly like the
notebook does, proving the Jupyter path runs end-to-end on a CPU box.
Skips YOLO *training* (needs ultralytics+GPU/time) but covers every
other notebook stage: config, data-gen, spectrogram, extraction,
classifier train/eval, confusion matrix.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.config import get_default_config, set_seed, get_device
from src.data_loader import build_synthetic_wideband, build_narrowband_classifier_dataset
from src.spectrogram import compute_spectrogram, spectrogram_to_image
from src.signal_detection import DetectedSignal
from src.signal_extraction import extract_narrowband
from src.train_classifier import load_narrowband_manifest, stratified_split
from src.classifier import build_model, train_classifier, TrainConfig, predict_classifier
from src.evaluate import evaluate_classifier
from src.visualization import (plot_iq, plot_spectrogram, plot_training_curves,
                               plot_confusion_matrix)
import matplotlib.pyplot as plt
import tempfile, shutil

cfg = get_default_config()
cfg.wideband_train, cfg.wideband_val, cfg.wideband_test = 8, 2, 2
cfg.narrowband_per_class = 12
set_seed(cfg.seed)
print("device:", get_device())
assert get_device().type == "cpu", "this simulation targets CPU"

tmp = Path(tempfile.mkdtemp(prefix="wsr_jupyter_"))
fig = Path("results/figures")
fig.mkdir(parents=True, exist_ok=True)

# §9-10: preprocessing + spectrogram + figure
sample = build_synthetic_wideband(cfg, split="test")[0]
spec = compute_spectrogram(sample.samples, fs=sample.fs, n_fft=cfg.n_fft,
                           hop_length=cfg.hop_length)
img = spectrogram_to_image(spec, image_size=cfg.image_size)
assert img.shape == (cfg.image_size, cfg.image_size, 3), img.shape
f, axes = plt.subplots(2, 1, figsize=(8, 7))
plot_iq(sample.samples, fs=sample.fs, ax=axes[0], n=2048)
plot_spectrogram(spec.db, spec.freq_hz, spec.time_s, ax=axes[1])
f.tight_layout()
f.savefig(fig / "01_spectrogram.png", dpi=80, bbox_inches="tight")
plt.close(f)
print("spectrogram + fig OK:", spec.shape, img.shape)

# §18: extraction with GT-derived boxes (stand-in for YOLO on CPU)
dets = []
for i, nbs in enumerate(sample.signals):
    dets.append(DetectedSignal(
        signal_id=i + 1, time_range=(float(spec.time_s[0]), float(spec.time_s[-1])),
        frequency_range=(nbs.carrier_offset_hz - 6000, nbs.carrier_offset_hz + 6000),
        bounding_box_yolo=(0.5, 0.5, 0.06, 1.0), confidence=0.99))
clips = extract_narrowband(sample.samples, fs=sample.fs, detections=dets,
                           target_len=cfg.clf_seq_len, cfg=cfg)
assert len(clips) == len(sample.signals), (len(clips), len(sample.signals))
assert clips[0].classifier_input.shape == (2, cfg.clf_seq_len)
print("extraction OK:", len(clips), "clips")

# §19-21: narrowband set + classifier train (tiny) + curves figure
counts = build_narrowband_classifier_dataset(str(tmp / "nb"), cfg=cfg)
X, y, classes = load_narrowband_manifest(str(tmp / "nb"))
tr, va, te = stratified_split(y, train=0.7, val=0.15, seed=cfg.seed)
model = build_model("cnn1d", num_classes=len(classes), in_channels=2,
                    seq_len=cfg.clf_seq_len, use_transfer=False)
tcfg = TrainConfig(epochs=2, batch_size=16, lr=1e-3, patience=2,
                   use_augmentation=True, use_scheduler=False,
                   device=str(get_device()))
hist = train_classifier(model, X[tr], y[tr], X[va], y[va], cfg=tcfg,
                        num_classes=len(classes), verbose=True)
f2 = plot_training_curves(hist, out_path=str(fig / "04_clf_training.png"))
plt.close(f2)
preds, _ = predict_classifier(model, X[te], device=get_device())
m = evaluate_classifier(y[te], preds, classes)
print("classifier test acc: %.3f macroF1: %.3f" % (m["accuracy"], m["f1_macro"]))

# §24: confusion matrix figure
f3 = plot_confusion_matrix(np.array(m["confusion_matrix"]), classes, normalize=True,
                           out_path=str(fig / "05_confusion_matrix.png"))
plt.close(f3)
shutil.rmtree(tmp, ignore_errors=True)
print("JUPYTER LOCAL SIMULATION OK; figures in", fig)
