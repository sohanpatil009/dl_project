"""Build the Kaggle notebook for the wideband signal recognition project.

Run once to regenerate `notebooks/wideband_signal_recognition_kaggle.ipynb`.

The Kaggle notebook mirrors the Colab notebook but:
* uses /kaggle/working/... for all outputs,
* explains how to attach the dataset via + Add Data,
* never assumes /content/drive exists,
* installs dependencies with %pip.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_notebook import md, code, nb_cells, write_notebook


SECTIONS: list = []

SECTIONS.append(md("""
# Wideband Signal Recognition — Kaggle Notebook (GPU)

> A. Vagollari, M. Hirschbeck, W. Gerstacker,
> **"An End-to-End Deep Learning Framework for Wideband Signal Recognition"**,
> IEEE Access, vol. 11, pp. 52899–52922, 2023.
> DOI: [10.1109/ACCESS.2023.3280454](https://doi.org/10.1109/ACCESS.2023.3280454)

Two-stage pipeline: **YOLOv5 detection** on wideband spectrograms, then
**1-D CNN / ResNet modulation classification** per detection.

## How to use this notebook on Kaggle

1. Create a new Kaggle Notebook with **GPU (T4 x2 or P100)** enabled:
   `Settings → Accelerator → GPU`.
2. Either run the dataset-generation cells below (works out of the box), or
   attach your own data:
   - Click **+ Add Data → Upload** (or link a Dataset) containing a
     `yolo_dataset/` folder with `images/{train,val,test}`, `labels/...`,
     and `data.yaml`, plus optionally a `narrowband/` folder.
   - The attached data appears under `/kaggle/input/<your-dataset>/`.
   - Set `USE_KAGGLE_INPUT = True` and `KAGGLE_INPUT_DIR` in the
     Configuration cell.
3. **Internet must be ON** the first time (to install `ultralytics` and
   download `yolov5s.pt`). Afterwards you can work offline.
4. All outputs go to `/kaggle/working/` (persistent only if you save the
   notebook version / download files).
"""))

SECTIONS.append(md("""
## 1. Environment check (Kaggle paths + GPU)
"""))
SECTIONS.append(code("""
import os, sys, platform
print('Python :', platform.python_version())
try:
    import torch
    print('PyTorch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
except Exception as e:
    print('PyTorch import failed:', e)
print('CWD:', os.getcwd())
print('Kaggle input:', os.listdir('/kaggle/input') if os.path.exists('/kaggle/input') else '(no /kaggle/input)')
print('Kaggle working:', os.listdir('/kaggle/working') if os.path.exists('/kaggle/working') else '(no /kaggle/working)')
"""))

SECTIONS.append(md("""
## 2. Get the project code

On Kaggle you have two options:
* **Option A (recommended):** clone from GitHub (needs Internet ON).
* **Option B:** upload this repo as a Kaggle Dataset and attach it as input.
"""))
SECTIONS.append(code("""
import os, subprocess
# Option A: clone from GitHub (IPython-safe: no indented shell escapes)
if not os.path.exists('/kaggle/working/wideband-signal-recognition'):
    subprocess.run(['git', 'clone',
                    'https://github.com/<your-user>/wideband-signal-recognition.git',
                    '/kaggle/working/wideband-signal-recognition'], check=False)
os.chdir('/kaggle/working/wideband-signal-recognition')
print(os.getcwd())
print(os.listdir('.'))
"""))

SECTIONS.append(md("""
## 3. Install dependencies (Kaggle)

Kaggle images already ship torch/torchvision. We only add what is missing.
"""))
SECTIONS.append(code("""
%pip install -q --no-deps -r requirements.txt
%pip install -q "ultralytics>=8.0.0" "opencv-python-headless>=4.6"
"""))

SECTIONS.append(md("""
## 4. Imports (Kaggle paths)
"""))
SECTIONS.append(code("""
import os, sys, json, time, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

ROOT = Path('/kaggle/working/wideband-signal-recognition')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config import get_default_config, set_seed, get_device
from src.preprocessing import generate_wideband, generate_narrowband
from src.spectrogram import compute_spectrogram, spectrogram_to_image
from src.data_loader import (
    build_synthetic_wideband, build_and_persist_yolo_dataset,
    build_narrowband_classifier_dataset, build_snr_sweep,
    discover_external_dataset,
)
from src.yolo_utils import install_yolov5, train_yolov5, load_yolov5, run_yolov5_inference, yolo_box_to_freq_time
from src.signal_detection import detect_signals
from src.signal_detection import DetectedSignal
from src.signal_extraction import extract_narrowband
from src.classifier import build_model, train_classifier, TrainConfig, predict_classifier
from src.evaluate import evaluate_detector, evaluate_classifier, snr_sweep_detector, snr_sweep_classifier
from src.inference import run_end_to_end_inference, results_to_dict
from src.visualization import plot_iq, plot_spectrogram, draw_boxes, plot_training_curves, plot_confusion_matrix, plot_metric_vs_snr

WORK = Path('/kaggle/working/wsr')
FIG = WORK / 'figures'; MET = WORK / 'metrics'; PRE = WORK / 'predictions'
for d in (FIG, MET, PRE):
    d.mkdir(parents=True, exist_ok=True)
print('Imports OK. Outputs ->', WORK)
"""))

SECTIONS.append(md("""
## 5. Configuration (Kaggle)

* `DATA_MODE='synthetic'` → built-in generator (works everywhere, no download).
* `DATA_MODE='radioml'` → REAL internet data: auto-finds an attached
  `RML2016.10a` mirror under `/kaggle/input/...`, else downloads the Zenodo
  mirror (~213 MB, needs Internet ON). Both stages use real clips.
* `DATA_MODE='attached'` → use a pre-built `yolo_dataset/` you attached via
  **+ Add Data**. Set `KAGGLE_INPUT_DIR=/kaggle/input/<your-dataset>`.
* `IMAGE_FORMAT`: `'jpg'` (smallest) | `'png'` (lossless) | `'bmp'`.
"""))
SECTIONS.append(code("""
DATA_MODE = 'synthetic'   # 'synthetic' | 'radioml' | 'attached'
KAGGLE_INPUT_DIR = '/kaggle/input/<your-dataset>'  # only for DATA_MODE='attached'
IMAGE_FORMAT = 'jpg'      # 'jpg' | 'png' | 'bmp'
USE_KAGGLE_INPUT = (DATA_MODE == 'attached')

OVERRIDES = {
    "wideband_train": 300,
    "wideband_val": 60,
    "wideband_test": 60,
    "narrowband_per_class": 200,
    "yolo_epochs": 20,
    "yolo_batch_size": 16,
    "yolo_input_size": 640,
    "clf_epochs": 25,
    "clf_batch_size": 128,
    "clf_arch": "resnet18",
    "clf_use_transfer": True,
    "clf_use_augmentation": True,
}
cfg = get_default_config()
for k, v in OVERRIDES.items():
    setattr(cfg, k, v)
set_seed(cfg.seed)
print('Device:', get_device())

if USE_KAGGLE_INPUT:
    found = discover_external_dataset(KAGGLE_INPUT_DIR)
    print('External dataset:', found)
    assert found is not None, f'No YOLO dataset found in {KAGGLE_INPUT_DIR}. Check the layout in data/README.md.'
    YOLO_DATASET_DIR = Path(found['images']).parents[0]
else:
    YOLO_DATASET_DIR = WORK / 'yolo_dataset'
NARROWBAND_DIR = WORK / 'narrowband'
print('YOLO dir:', YOLO_DATASET_DIR)
print('Narrowband dir:', NARROWBAND_DIR)
"""))

SECTIONS.append(md("""
## 6. Dataset setup (generate or attach)
"""))
SECTIONS.append(code("""
cfg.image_format = IMAGE_FORMAT
if DATA_MODE == 'radioml':
    cfg.data_source = 'radioml'
    from src.data_loader import load_real_pool
    from src.public_data import find_kaggle_radioml
    pkl = find_kaggle_radioml()
    print('Kaggle RadioML mirror:', pkl or '(none attached - will download ~213 MB from Zenodo)')
    real_pool = load_real_pool(cfg, pkl_path=pkl, dest_dir=str(WORK / 'public'))
    print('Real pool:', {k: len(v) for k, v in real_pool.items()})
    import shutil
    for d in (YOLO_DATASET_DIR, NARROWBAND_DIR):
        if d.exists():
            shutil.rmtree(d)
    splits = build_and_persist_yolo_dataset(out_dir=str(YOLO_DATASET_DIR), cfg=cfg,
                                            num_classes=1, real_pool=real_pool)
    for k, v in splits.items():
        print(f'  {k}: {len(v)} real-mix images (.{cfg.image_format})')
    counts = build_narrowband_classifier_dataset(out_dir=str(NARROWBAND_DIR), cfg=cfg,
                                                 real_pool=real_pool)
    print('  per-class (real):', counts)
elif not USE_KAGGLE_INPUT:
    if not (YOLO_DATASET_DIR / 'data.yaml').exists():
        print('Generating YOLO dataset ...')
        splits = build_and_persist_yolo_dataset(out_dir=str(YOLO_DATASET_DIR), cfg=cfg, num_classes=1)
        for k, v in splits.items():
            print(f'  {k}: {len(v)} images')
    else:
        print('YOLO dataset already present.')
    if not (NARROWBAND_DIR / 'manifest.json').exists():
        print('Generating narrowband clips ...')
        counts = build_narrowband_classifier_dataset(out_dir=str(NARROWBAND_DIR), cfg=cfg)
        print('  per-class:', counts)
    else:
        print('Narrowband dataset already present.')
else:
    print('Using attached Kaggle input at', KAGGLE_INPUT_DIR)
    print(open(Path(KAGGLE_INPUT_DIR) / 'data.yaml').read() if (Path(KAGGLE_INPUT_DIR) / 'data.yaml').exists() else '(no data.yaml in input root)')
    # Attached mode only provides the YOLO set; still need narrowband clips
    # for the classifier stage -> generate them (synthetic is fine here).
    if not (NARROWBAND_DIR / 'manifest.json').exists():
        print('Building narrowband clips (synthetic) for the classifier ...')
        counts = build_narrowband_classifier_dataset(out_dir=str(NARROWBAND_DIR), cfg=cfg)
        print('  per-class:', counts)
"""))

SECTIONS.append(md("""
## 7. Dataset inspection
"""))
SECTIONS.append(code("""
import json
from glob import glob
from collections import Counter
for sub in ('train', 'val', 'test'):
    n_img = sum(len(glob(str(YOLO_DATASET_DIR / 'images' / sub / f'*.{e}')))
                for e in ('jpg', 'jpeg', 'png', 'bmp'))
    n_lbl = len(glob(str(YOLO_DATASET_DIR / 'labels' / sub / '*.txt')))
    print(f'{sub}: {n_img} images, {n_lbl} labels')
manifest = json.load(open(NARROWBAND_DIR / 'manifest.json')) if (NARROWBAND_DIR / 'manifest.json').exists() else []
print('narrowband clips:', len(manifest), dict(Counter(c for c, _ in manifest)) if manifest else '')
print('\\ndata.yaml:')
print(open(YOLO_DATASET_DIR / 'data.yaml').read())
"""))

SECTIONS.append(md("""
## 8. Preprocessing + spectrogram example
"""))
SECTIONS.append(code("""
sample = build_synthetic_wideband(cfg, split='test')[0]
print('signals:', [(s.modulation, round(s.carrier_offset_hz)) for s in sample.signals])
spec = compute_spectrogram(sample.samples, fs=sample.fs, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
img = spectrogram_to_image(spec, image_size=cfg.image_size)
print('spec:', spec.shape, 'img:', img.shape)
fig, axes = plt.subplots(2, 1, figsize=(8, 7))
plot_iq(sample.samples, fs=sample.fs, ax=axes[0], n=2048)
plot_spectrogram(spec.db, spec.freq_hz, spec.time_s, ax=axes[1])
fig.tight_layout(); fig.savefig(FIG / '01_spectrogram.png', dpi=120, bbox_inches='tight'); plt.show()
"""))

SECTIONS.append(md("""
## 9. Train YOLOv5 detector
"""))
SECTIONS.append(code("""
install_yolov5()
yolo_weights = train_yolov5(
    data_yaml=str(YOLO_DATASET_DIR / 'data.yaml'),
    cfg=cfg, weights='yolov5s.pt',
    project=str(WORK / 'runs'), name='detector',
)
print('Best weights ->', yolo_weights)
"""))

SECTIONS.append(md("""
## 10. Validate detector + visualise predictions
"""))
SECTIONS.append(code("""
from ultralytics import YOLO
from PIL import Image
import glob
model = YOLO(str(yolo_weights))
val_results = model.val(data=str(YOLO_DATASET_DIR / 'data.yaml'), imgsz=cfg.yolo_input_size,
                        batch=cfg.yolo_batch_size, device='0' if get_device().type == 'cuda' else 'cpu',
                        project=str(WORK / 'runs'), name='detector_val', exist_ok=True)
print('P/R/mAP50/mAP =', float(val_results.box.mp), float(val_results.box.mr), float(val_results.box.map50), float(val_results.box.map))
import itertools
test_imgs = sorted(itertools.chain.from_iterable(
    glob.glob(str(YOLO_DATASET_DIR / 'images' / 'test' / f'*.{e}'))
    for e in ('jpg', 'jpeg', 'png', 'bmp')))[:8]
det_lists = run_yolov5_inference(model, test_imgs, cfg=cfg, conf=0.25, iou=0.45)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, path, dets in zip(axes.flat, test_imgs, det_lists):
    ax.imshow(np.asarray(Image.open(path).convert('RGB')))
    draw_boxes(ax, [(d.cx, d.cy, d.w, d.h) for d in dets],
               labels=[f'{d.class_name} {d.confidence:.2f}' for d in dets], color='red')
    ax.set_title(Path(path).name); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(FIG / '03_detector_predictions.png', dpi=120, bbox_inches='tight'); plt.show()
"""))

SECTIONS.append(md("""
## 11. Train modulation classifier (CNN / ResNet)
"""))
SECTIONS.append(code("""
from src.train_classifier import load_narrowband_manifest, stratified_split
import torch as _torch
X, y, classes = load_narrowband_manifest(str(NARROWBAND_DIR))
print('X:', X.shape, 'classes:', classes)
tr, va, te = stratified_split(y, train=0.7, val=0.15, seed=cfg.seed)
model_clf = build_model(cfg.clf_arch, num_classes=len(classes), in_channels=2,
                        seq_len=cfg.clf_seq_len, use_transfer=cfg.clf_use_transfer)
tcfg = TrainConfig(epochs=cfg.clf_epochs, batch_size=cfg.clf_batch_size, lr=cfg.clf_lr,
                   weight_decay=cfg.clf_weight_decay, patience=cfg.clf_patience,
                   use_augmentation=cfg.clf_use_augmentation, use_scheduler=True, device=str(get_device()))
hist = train_classifier(model_clf, X[tr], y[tr], X[va], y[va], cfg=tcfg, num_classes=len(classes), verbose=True)
_torch.save({'state_dict': model_clf.state_dict(), 'arch': cfg.clf_arch, 'classes': classes, 'config': cfg.to_dict()},
            str(WORK / 'classifier.pt'))
fig = plot_training_curves(hist, out_path=str(FIG / '04_clf_training.png')); plt.show()
preds, probs = predict_classifier(model_clf, X[te], device=get_device())
metrics = evaluate_classifier(y[te], preds, classes, out_dir=str(MET))
print('test acc:', metrics['accuracy'], 'macro F1:', metrics['f1_macro'])
fig = plot_confusion_matrix(__import__('numpy').array(metrics['confusion_matrix']), classes, normalize=True,
                            out_path=str(FIG / '05_confusion_matrix.png')); plt.show()
"""))

SECTIONS.append(md("""
## 12. End-to-end inference on one capture
"""))
SECTIONS.append(code("""
sample = build_synthetic_wideband(cfg, split='test')[0]
spec, dets, clips, results = run_end_to_end_inference(
    sample.samples, sample.fs, detector_weights=str(yolo_weights),
    classifier_ckpt=str(WORK / 'classifier.pt'), cfg=cfg)
for r in results:
    print(f'signal {r.signal_id}: f=[{r.frequency_range_hz[0]:+.0f},{r.frequency_range_hz[1]:+.0f}] Hz '
          f'mod={r.predicted_modulation} det={r.detector_confidence:.2f} clf={r.classifier_confidence:.2f}')
with open(PRE / 'end_to_end.json', 'w') as f:
    json.dump(results_to_dict(results), f, indent=2)
print('saved ->', PRE / 'end_to_end.json')
"""))

SECTIONS.append(md("""
## 13. SNR sweep (-5 … 20 dB) + paper comparison

Paper (approx.): detector P≈0.77 / R≈0.82 / mAP≈0.86; classifier ≈0.60 @0dB → ≈0.95 @20dB.
Our numbers come from the synthetic set — report them as-is, do not force a match.
"""))
SECTIONS.append(code("""
import csv
det_sweep = snr_sweep_detector(model, cfg=cfg, per_snr=15)
clf_sweep = snr_sweep_classifier(model_clf, classes, cfg=cfg, per_snr_per_class=15)
snrs = list(det_sweep.keys())
with open(MET / 'snr_sweep.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['SNR_dB', 'Det_P', 'Det_R', 'Det_IoU', 'Clf_Acc', 'Clf_F1'])
    for s in snrs:
        w.writerow([s, f"{det_sweep[s]['precision']:.3f}", f"{det_sweep[s]['recall']:.3f}",
                    f"{det_sweep[s]['mean_iou']:.3f}", f"{clf_sweep[s]['accuracy']:.3f}",
                    f"{clf_sweep[s]['f1_macro']:.3f}"])
plot_metric_vs_snr(snrs, {'Det-P': [det_sweep[s]['precision'] for s in snrs],
                          'Det-R': [det_sweep[s]['recall'] for s in snrs]},
                   ylabel='Detector', out_path=str(FIG / '06_det_vs_snr.png')); plt.show()
plot_metric_vs_snr(snrs, {'Clf-Acc': [clf_sweep[s]['accuracy'] for s in snrs],
                          'Clf-F1': [clf_sweep[s]['f1_macro'] for s in snrs]},
                   ylabel='Classifier', out_path=str(FIG / '07_clf_vs_snr.png')); plt.show()
import pandas as pd
paper = {'Det-P': 0.77, 'Det-R': 0.82, 'mAP50': 0.86, 'Clf@20dB': 0.95, 'Clf@0dB': 0.60}
ours = {'Det-P': float(val_results.box.mp), 'Det-R': float(val_results.box.mr),
        'mAP50': float(val_results.box.map50),
        'Clf@20dB': clf_sweep[20.0]['accuracy'] if 20.0 in clf_sweep else float('nan'),
        'Clf@0dB': clf_sweep[0.0]['accuracy'] if 0.0 in clf_sweep else float('nan')}
print(pd.DataFrame({'paper': paper, 'ours': ours}).round(3).to_markdown())
"""))

SECTIONS.append(md("""
## 14. Limitations, improvements, conclusion

**Limitations:** original dataset not public → numbers are on fresh synthetic
data; no hardware impairments modelled; ImageNet transfer is partial (first
conv re-trained); default YOLO thresholds (conf=0.25, IoU=0.45).

**Improvements:** real OTA captures; larger YOLO variants; per-modulation YOLO
classes; mixup/SpecAugment; joint detector+classifier loss.

**Conclusion:** full two-stage YOLOv5 + ResNet pipeline runs end-to-end on
Kaggle GPU. Differences vs the paper are expected and documented — do not
fabricate a match.
"""))


nb = nb_cells(*SECTIONS)
# Kaggle metadata: keep kernelspec python3, add kaggle-specific accelerator tag
nb["metadata"]["kaggle"] = {"accelerator": "GPU", "language": "python", "isInternetEnabled": True}
out = Path(__file__).resolve().parents[1] / 'notebooks' / 'wideband_signal_recognition_kaggle.ipynb'
write_notebook(str(out), nb)
print('Wrote', out)
