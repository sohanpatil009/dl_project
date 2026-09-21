"""Live run: classifier SNR sweep + figures + mock end-to-end demo (CPU)."""
import sys, json, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.config import get_default_config, get_device
from src.classifier import build_model, predict_classifier
from src.evaluate import evaluate_classifier, snr_sweep_classifier
from src.visualization import plot_training_curves, plot_confusion_matrix, plot_metric_vs_snr
from src.data_loader import build_synthetic_wideband
from src.preprocessing import generate_narrowband
from src.spectrogram import compute_spectrogram
from src.signal_detection import DetectedSignal
from src.signal_extraction import extract_narrowband
from src.inference import results_to_dict
from src.train_classifier import load_narrowband_manifest, stratified_split

cfg = get_default_config()
device = get_device()
FIG = Path("results/figures"); MET = Path("results/metrics"); PRE = Path("results/predictions")
for d in (FIG, MET, PRE):
    d.mkdir(parents=True, exist_ok=True)

ckpt = torch.load("models/classifier_run.pt", map_location=device, weights_only=False)
model = build_model(arch=ckpt["arch"], num_classes=len(ckpt["classes"]),
                    in_channels=2, seq_len=cfg.clf_seq_len, use_transfer=False)
model.load_state_dict(ckpt["state_dict"])
classes = ckpt["classes"]
print("model:", ckpt["arch"], "classes:", classes)

# 1. Held-out test eval + confusion matrix
X, y, _ = load_narrowband_manifest("data/narrowband")
_, _, te = stratified_split(y, train=0.7, val=0.15, seed=cfg.seed)
preds, _ = predict_classifier(model, X[te], device=device)
m = evaluate_classifier(y[te], preds, classes, out_dir=str(MET))
print("TEST acc=%.4f macroF1=%.4f" % (m["accuracy"], m["f1_macro"]))

# 2. Training curves from saved history
hist = json.load(open(MET / "clf_history_run.json"))["history"]
f = plot_training_curves(hist, out_path=str(FIG / "04_clf_training.png"))
plt.close(f)

# 3. SNR sweep (small for CPU)
sweep = snr_sweep_classifier(model, classes, cfg=cfg, per_snr_per_class=12)
snrs = sorted(sweep)
acc = [sweep[s]["accuracy"] for s in snrs]
f1 = [sweep[s]["f1_macro"] for s in snrs]
with open(MET / "snr_sweep_run.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["SNR_dB", "Clf_Acc", "Clf_F1"])
    for s, a, v in zip(snrs, acc, f1):
        w.writerow([s, "%.3f" % a, "%.3f" % v])
f = plot_metric_vs_snr(snrs, {"Clf-Acc": acc, "Clf-F1": f1},
                       ylabel="Classifier metrics",
                       out_path=str(FIG / "07_clf_vs_snr.png"))
plt.close(f)
print("SNR sweep:")
for s, a, v in zip(snrs, acc, f1):
    print("  %5.0f dB  acc=%.3f  F1=%.3f" % (s, a, v))

# 4. Mock end-to-end on 3 wideband test samples (GT boxes stand in for YOLO)
samples = build_synthetic_wideband(cfg, split="test")[:3]
demo = []
for i, s in enumerate(samples):
    spec = compute_spectrogram(s.samples, fs=s.fs, n_fft=cfg.n_fft,
                               hop_length=cfg.hop_length)
    dets = [DetectedSignal(
        signal_id=k + 1,
        time_range=(float(spec.time_s[0]), float(spec.time_s[-1])),
        frequency_range=(nbs.carrier_offset_hz - 6000, nbs.carrier_offset_hz + 6000),
        bounding_box_yolo=(0.5, 0.5, 0.06, 1.0), confidence=0.99)
        for k, nbs in enumerate(s.signals)]
    clips = extract_narrowband(s.samples, fs=s.fs, detections=dets,
                               target_len=cfg.clf_seq_len, cfg=cfg)
    Xq = np.stack([c.classifier_input for c in clips], axis=0)
    p, pr = predict_classifier(model, Xq, device=device)
    recs = [{"signal_id": int(c.signal_id),
             "frequency_range": [float(c.frequency_range[0]), float(c.frequency_range[1])],
             "time_range": [float(c.time_range[0]), float(c.time_range[1])],
             "detector_confidence": 0.99,
             "predicted_modulation": classes[int(p[k])],
             "classifier_confidence": float(pr[k].max()),
             "true_modulation": s.signals[k].modulation}
            for k, c in enumerate(clips)]
    demo.append({"sample_id": i, "results": recs})
    for r in recs:
        print("sample %d sig %d: true=%-7s pred=%-7s conf=%.2f" % (
            i, r["signal_id"], r["true_modulation"],
            r["predicted_modulation"], r["classifier_confidence"]))
with open(PRE / "end_to_end_run.json", "w") as fh:
    json.dump(demo, fh, indent=2)
print("LIVE RUN OK")
