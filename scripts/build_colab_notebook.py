"""Build the Google Colab notebook for the wideband signal recognition
project.

Run this once to regenerate `notebooks/wideband_signal_recognition_colab.ipynb`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_notebook import md, code, nb_cells, write_notebook


SECTIONS: list = []


# ---------------------------------------------------------------------------
# 1. Project introduction
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
# Wideband Signal Recognition — End-to-End Deep Learning

This notebook reproduces / adapts the work of:

> A. Vagollari, M. Hirschbeck, W. Gerstacker,
> **"An End-to-End Deep Learning Framework for Wideband Signal Recognition"**,
> IEEE Access, vol. 11, pp. 52899–52922, 2023.
> DOI: [10.1109/ACCESS.2023.3280454](https://doi.org/10.1109/ACCESS.2023.3280454)

It implements a two-stage deep-learning pipeline:

1. **YOLOv5** detects individual narrowband signals in a wideband
   spectrogram.
2. A **1-D CNN / ResNet** classifier recognises the modulation
   (BPSK, QPSK, 8PSK, 16QAM, 64QAM, NOISE, NO_SIGNAL) of each
   detected signal.

The notebook is organised into the 29 sections listed in the project
specification. Every cell can be executed in order; the project also
runs on Kaggle and locally.
"""))


# ---------------------------------------------------------------------------
# 2. Paper info
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 2. Paper Information

| Field | Value |
|---|---|
| Title | An End-to-End Deep Learning Framework for Wideband Signal Recognition |
| Authors | Adela Vagollari, Martin Hirschbeck, Wolfgang Gerstacker |
| Venue | IEEE Access, vol. 11, pp. 52899-52922, 2023 |
| DOI | 10.1109/ACCESS.2023.3280454 |

The paper describes a system that **first detects** signals in the
spectrogram of a wideband capture using YOLOv5s and **then classifies**
each detected narrowband signal with a 1-D CNN. The reported
performance is roughly:

* Detector: Precision ≈ 77 %, Recall ≈ 82 %, mAP@0.5 ≈ 86 %
* Classifier: accuracy scales from ≈ 60 % at 0 dB to ≈ 95 % at 20 dB SNR.

The original synthetic dataset is **not publicly available**; this
implementation therefore *generates a comparable dataset from scratch*
following the parameters described in the paper (Section III).
"""))


# ---------------------------------------------------------------------------
# 3. Environment setup
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 3. Environment Setup

Run the following cell to set the working directory to the project
root (cloned from GitHub) and to check the Python / GPU environment.
"""))
SECTIONS.append(code("""
import os, sys, platform
print('Python :', platform.python_version())
print('System :', platform.system(), platform.release())
try:
    import torch
    print('PyTorch:', torch.__version__)
    print('CUDA   :', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
except Exception as e:
    print('PyTorch not installed:', e)

# --- Locate the project root (works for ALL Colab upload methods) ---
# Option A: you cloned a git repo  -> /content/wideband-signal-recognition
# Option B: you uploaded a .zip and unzipped it -> /content/<your-folder>
# Option C: you mounted Drive and uploaded there
# We simply search upwards from CWD for `src/config.py`.
from pathlib import Path as _P
def _find_project_root(start='.', markers=('src/config.py', 'requirements.txt')):
    start = _P(start).resolve()
    for cand in [start] + list(start.parents):
        if all((cand / m).exists() for m in markers):
            return cand
        # also accept a single nested checkout, e.g. /content/repo/<project>
        try:
            subs = sorted([p for p in cand.iterdir() if p.is_dir()])
        except Exception:
            continue
        for sub in subs:
            try:
                if all((sub / m).exists() for m in markers):
                    return sub
            except Exception:
                continue
    return None
ROOT = _find_project_root('.')
if ROOT is None:
    # Last resort: optional git clone. EDIT THIS URL to your own fork,
    # or upload the project zip manually (see guide below) and re-run.
    _REPO_URL = 'https://github.com/sohanpatil009/dl_project.git'
    _DEST = '/content/dl_project'
    print(f'Project not found. Attempting git clone from {_REPO_URL} ...')
    print('If this fails (404), upload the project folder/zip to /content and re-run this cell.')
    import subprocess
    r = subprocess.run(['git', 'clone', _REPO_URL, _DEST], check=False)
    ROOT = _find_project_root(_DEST) or _find_project_root('/content')
if ROOT is None:
    raise RuntimeError(
        'Cannot find project root (expected src/config.py + requirements.txt). '
        'Upload the project folder to /content (e.g. via Files > Upload, unzip if needed) '
        'then re-run this cell. Current CWD: ' + os.getcwd() +
        ' | /content: ' + str(sorted(os.listdir(\"/content\")) if os.path.exists(\"/content\") else \"n/a\"))
os.chdir(str(ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print('CWD (project root):', os.getcwd())

# Optional: mount Google Drive ONLY if you need files from Drive.
# Disabled by default because drive.mount() blocks on an auth code.
MOUNT_DRIVE = False
if MOUNT_DRIVE:
    try:
        from google.colab import drive  # type: ignore
        drive.mount('/content/drive')
        print('Drive mounted at /content/drive')
    except Exception as e:
        print('Drive mount skipped/failed:', e)
else:
    print('Drive mount skipped (set MOUNT_DRIVE=True to enable).')
"""))


# ---------------------------------------------------------------------------
# 4. Dependency installation
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 4. Dependency Installation
"""))
SECTIONS.append(code("""
# Colab already ships torch/torchvision + CUDA + jupyter. A plain
# `pip install -r requirements.txt` would REINSTALL torch (~2 GB) and
# downgrade numpy, breaking the runtime. So we filter those out.
import importlib.util
def _have(pkg): return importlib.util.find_spec(pkg) is not None
print('torch present:', _have('torch'), '| torchvision present:', _have('torchvision'))
_SKIPPED = ('torch', 'torchvision', 'ipython', 'ipykernel', 'jupyter', 'notebook', 'nbclient', 'nbformat')
kept = []
for line in open('requirements.txt'):
    s = line.strip()
    if not s or s.startswith('#'):
        continue
    name = s.split('>')[0].split('<')[0].split('=')[0].split('[')[0].strip().lower().replace('_', '-')
    if name in _SKIPPED and _have(name.split('-')[0] if '-' not in name else name):
        print('  skip (preinstalled):', s)
        continue
    if name in ('torch', 'torchvision') and _have(name):
        print('  skip (preinstalled):', s)
        continue
    kept.append(s)
open('/tmp/requirements-colab.txt', 'w').write('\\n'.join(kept) + '\\n')
print('Installing:', kept)
!pip install -q -r /tmp/requirements-colab.txt
!pip install -q \"ultralytics>=8.0.0\" \"opencv-python-headless>=4.6\"
# Sanity check
import numpy, scipy, sklearn, PIL
print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| PIL', PIL.__version__)
try:
    import torch, ultralytics
    print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '| ultralytics OK')
except Exception as e:
    print('Import check FAILED:', e)
    print('If torch broke, Runtime > Restart runtime and re-run only this cell.')
"""))


# ---------------------------------------------------------------------------
# 5. Imports
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 5. Imports
"""))
SECTIONS.append(code("""
import os, sys, json, time, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

# Make sure the `src` package is importable
ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config import get_default_config, set_seed, get_device, FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR
from src.preprocessing import generate_wideband, generate_narrowband
from src.spectrogram import compute_spectrogram, spectrogram_to_image
from src.data_loader import (
    build_synthetic_wideband,
    build_and_persist_yolo_dataset,
    build_narrowband_classifier_dataset,
    build_snr_sweep,
)
from src.dataset_utils import save_wideband_dataset, write_yolo_yaml
from src.yolo_utils import install_yolov5, train_yolov5, load_yolov5, run_yolov5_inference, yolo_box_to_freq_time
from src.signal_detection import detect_signals
from src.signal_extraction import extract_narrowband
from src.classifier import build_model, train_classifier, TrainConfig, predict_classifier
from src.evaluate import (
    evaluate_detector, evaluate_classifier,
    snr_sweep_detector, snr_sweep_classifier,
)
from src.inference import run_end_to_end_inference, results_to_dict
from src.visualization import (
    plot_iq, plot_spectrogram, draw_boxes,
    plot_training_curves, plot_confusion_matrix, plot_metric_vs_snr,
)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
print('All imports OK')
"""))


# ---------------------------------------------------------------------------
# 6. Configuration
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 6. Configuration

The whole project is driven by a single `Config` object. Edit the
`OVERRIDES` dict below to change any parameter without touching the
source code.
"""))
SECTIONS.append(code("""
OVERRIDES = {
    # Dataset sizes (small for fast iteration on Colab free tier)
    "wideband_train": 300,
    "wideband_val": 60,
    "wideband_test": 60,
    "narrowband_per_class": 200,
    # YOLO training
    "yolo_epochs": 20,
    "yolo_batch_size": 16,
    "yolo_input_size": 640,
    # Classifier
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
print('Config saved to', cfg.save())
"""))


# ---------------------------------------------------------------------------
# 7. Dataset download / setup
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 7. Dataset Download / Setup

The original Vagollari et al. (2023) dataset is **not publicly
released**. We therefore *generate* a comparable synthetic dataset
following the parameters reported in the paper. If you have your own
captures, drop them in `data/yolo_dataset/` matching the YOLO layout
and skip the generation step.
"""))
SECTIONS.append(code("""
DATA_ROOT = Path('data')
YOLO_DATASET_DIR = DATA_ROOT / 'yolo_dataset'
NARROWBAND_DIR = DATA_ROOT / 'narrowband'
DATA_ROOT.mkdir(exist_ok=True)

# 1. Wideband -> spectrogram + YOLO labels
if not (YOLO_DATASET_DIR / 'data.yaml').exists():
    print('Generating YOLO dataset ...')
    splits = build_and_persist_yolo_dataset(out_dir=str(YOLO_DATASET_DIR), cfg=cfg, num_classes=1)
    for k, v in splits.items():
        print(f'  {k}: {len(v)} images')
else:
    print('YOLO dataset already present, skipping generation')

# 2. Narrowband clips for the classifier
if not (NARROWBAND_DIR / 'manifest.json').exists():
    print('Generating narrowband clips ...')
    counts = build_narrowband_classifier_dataset(out_dir=str(NARROWBAND_DIR), cfg=cfg)
    print('  per-class counts:', counts)
else:
    print('Narrowband dataset already present, skipping generation')
"""))


# ---------------------------------------------------------------------------
# 7b. Real internet data (optional, both stages)
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 7b. (Optional) Use REAL Internet Data Instead of Synthetic

Set `USE_REAL_DATA = True` to download **RadioML 2016.10A** (DeepSig,
CC BY-NC-SA 4.0, ~213 MB via Zenodo DOI `10.5281/zenodo.18397070`) and build
**both** stages from real clips: the classifier trains on real 2×128 I/Q,
and wideband spectrogram **images** are mixed from tiled real clips + YOLO
boxes. 5/7 classes overlap the paper exactly (BPSK/QPSK/8PSK/16QAM/64QAM);
NOISE/NO_SIGNAL are synthesised; train bin 18 dB ≈ paper's 20 dB.
Set `IMAGE_FORMAT` to `'jpg'` (smallest), `'png'` (lossless) or `'bmp'`.
"""))
SECTIONS.append(code("""
USE_REAL_DATA = False
IMAGE_FORMAT = 'jpg'   # 'jpg' | 'png' | 'bmp'
if USE_REAL_DATA:
    cfg.data_source = 'radioml'
    cfg.image_format = IMAGE_FORMAT
    from src.data_loader import load_real_pool
    print('Downloading/loading RadioML 2016.10A ...')
    real_pool = load_real_pool(cfg, dest_dir=str(DATA_ROOT / 'public'))
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
else:
    print('Using synthetic data (set USE_REAL_DATA=True for RadioML internet data).')
"""))


# ---------------------------------------------------------------------------
# 8. Dataset inspection
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 8. Dataset Inspection
"""))
SECTIONS.append(code("""
import json
from glob import glob

print('YOLO dataset contents:')
import itertools
for sub in ('train', 'val', 'test'):
    n_img = sum(len(glob(str(YOLO_DATASET_DIR / 'images' / sub / f'*.{e}')))
                for e in ('jpg', 'jpeg', 'png', 'bmp'))
    n_lbl = len(glob(str(YOLO_DATASET_DIR / 'labels' / sub / '*.txt')))
    print(f'  {sub}: {n_img} images, {n_lbl} labels')

print('\\nNarrowband dataset:')
manifest = json.load(open(NARROWBAND_DIR / 'manifest.json'))
from collections import Counter
print('  per-class counts:', dict(Counter(c for c, _ in manifest)))

print('\\ndata.yaml:')
print(open(YOLO_DATASET_DIR / 'data.yaml').read())
"""))


# ---------------------------------------------------------------------------
# 9. Data preprocessing
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 9. Data Preprocessing

The preprocessing pipeline synthesises wideband captures and
narrowband clips. The generators live in `src.preprocessing` and the
spectrogram conversion in `src.spectrogram`. Both are deterministic
when the seed is fixed.
"""))
SECTIONS.append(code("""
# Generate one example and inspect its structure
sample = build_synthetic_wideband(cfg, split='test')[0]
print('Sample:', len(sample.samples), 'IQ samples @', sample.fs, 'Hz')
print('Signals in capture:')
for s in sample.signals:
    print(f'  {s.modulation:8s}  f_c = {s.carrier_offset_hz:+8.0f} Hz  '
          f'(SNR target {s.snr_db} dB)')

# Save the preprocessing parameters to disk for reproducibility
(cfg.save())
"""))


# ---------------------------------------------------------------------------
# 10. Spectrogram generation
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 10. Spectrogram Generation

The wideband I/Q signal is converted to a 1024 x 1024 RGB image
suitable for YOLOv5. The pipeline is configurable in `src.spectrogram`.
"""))
SECTIONS.append(code("""
spec = compute_spectrogram(sample.samples, fs=sample.fs,
                           n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                           window=cfg.window, db_floor=cfg.db_floor)
img = spectrogram_to_image(spec, image_size=cfg.image_size)
print('Spectrogram shape :', spec.shape, '(freq, time)')
print('YOLO image shape  :', img.shape, '(H, W, C)')

fig, axes = plt.subplots(2, 1, figsize=(8, 7))
plot_iq(sample.samples, fs=sample.fs, ax=axes[0], n=2048, title='Wideband I/Q (first 2048 samples)')
plot_spectrogram(spec.db, spec.freq_hz, spec.time_s, ax=axes[1], title='Wideband spectrogram')
fig.tight_layout()
fig.savefig(FIGURES_DIR / '01_spectrogram.png', dpi=120, bbox_inches='tight')
plt.show()
"""))


# ---------------------------------------------------------------------------
# 11. Dataset visualization
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 11. Dataset Visualization

Let's visualise a few spectrograms with their ground-truth bounding
boxes to confirm the dataset is realistic.
"""))
SECTIONS.append(code("""
import random
random.seed(0)
samples = build_synthetic_wideband(cfg, split='val')[:4]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, s in zip(axes.flat, samples):
    spec = compute_spectrogram(s.samples, fs=s.fs, n_fft=cfg.n_fft,
                               hop_length=cfg.hop_length)
    plot_spectrogram(spec.db, spec.freq_hz, spec.time_s, ax=ax, title=None)
    boxes = []
    labels = []
    for nbs in s.signals:
        f_lo = float(spec.freq_hz.min()); f_hi = float(spec.freq_hz.max())
        t_lo = float(spec.time_s.min()); t_hi = float(spec.time_s.max())
        cx = (nbs.carrier_offset_hz - f_lo) / (f_hi - f_lo)
        cy = 0.5
        w = (cfg.bandwidth + 2*cfg.padding) / (f_hi - f_lo)
        h = 1.0
        boxes.append((cx, cy, w, h))
        labels.append(nbs.modulation)
    draw_boxes(ax, boxes, labels=labels, color='red', line_width=1.4)
fig.tight_layout()
fig.savefig(FIGURES_DIR / '02_dataset_examples.png', dpi=120, bbox_inches='tight')
plt.show()
"""))


# ---------------------------------------------------------------------------
# 12. Train/val/test split
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 12. Train / Validation / Test Split

The YOLO dataset is already split 70/15/15 by the build script. For
the narrowband dataset we perform a stratified split below.
"""))
SECTIONS.append(code("""
import json
manifest = json.load(open(NARROWBAND_DIR / 'manifest.json'))
from collections import defaultdict
by_class = defaultdict(list)
for cls, path in manifest:
    by_class[cls].append(path)
print('Class distribution:')
for c, items in by_class.items():
    print(f'  {c:10s} -> {len(items):4d} clips')

train, val, test = [], [], []
rng = random.Random(cfg.seed)
for c, items in by_class.items():
    rng.shuffle(items)
    n = len(items)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train.extend(items[:n_train])
    val.extend(items[n_train:n_train + n_val])
    test.extend(items[n_train + n_val:])
print(f'Split sizes: train={len(train)}, val={len(val)}, test={len(test)}')
"""))


# ---------------------------------------------------------------------------
# 13. YOLO dataset preparation
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 13. YOLO Dataset Preparation

The build script already wrote a `data.yaml` file. We re-print it
here and verify the directory layout.
"""))
SECTIONS.append(code("""
print(open(YOLO_DATASET_DIR / 'data.yaml').read())
import os
for sub in ('train', 'val', 'test'):
    img_dir = YOLO_DATASET_DIR / 'images' / sub
    lbl_dir = YOLO_DATASET_DIR / 'labels' / sub
    n_img = sum(len(list(img_dir.glob(f'*.{e}'))) for e in ('jpg', 'jpeg', 'png', 'bmp'))
    n_lbl = len(list(lbl_dir.glob('*.txt')))
    print(f'{sub}: images={n_img}, labels={n_lbl}')
"""))


# ---------------------------------------------------------------------------
# 14. YOLOv5 training
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 14. YOLOv5 Training

We use the Ultralytics YOLOv5s checkpoint as the starting point
(paper uses YOLOv5s). The training runs on the GPU if one is
available, otherwise on the CPU.
"""))
SECTIONS.append(code("""
install_yolov5()
yolo_weights = train_yolov5(
    data_yaml=str(YOLO_DATASET_DIR / 'data.yaml'),
    cfg=cfg,
    weights='yolov5s.pt',
    project='models/runs',
    name='detector',
)
print('Best weights ->', yolo_weights)
"""))


# ---------------------------------------------------------------------------
# 15. YOLOv5 validation
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 15. YOLOv5 Validation

Run Ultralytics' built-in `val()` to obtain precision, recall, mAP@0.5
and mAP@0.5:0.95 on the held-out split.
"""))
SECTIONS.append(code("""
from ultralytics import YOLO
model = YOLO(str(yolo_weights))
val_results = model.val(
    data=str(YOLO_DATASET_DIR / 'data.yaml'),
    imgsz=cfg.yolo_input_size,
    batch=cfg.yolo_batch_size,
    device='0' if get_device().type == 'cuda' else 'cpu',
    project='models/runs', name='detector_val', exist_ok=True,
)
print('Detector validation results:')
print('  Precision  :', float(val_results.box.mp))
print('  Recall     :', float(val_results.box.mr))
print('  mAP@0.5    :', float(val_results.box.map50))
print('  mAP@0.5:.95:', float(val_results.box.map))
"""))


# ---------------------------------------------------------------------------
# 16. Signal detection
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 16. Signal Detection

Load the trained model and run it on a few test spectrograms.
"""))
SECTIONS.append(code("""
import glob, itertools
test_imgs = sorted(itertools.chain.from_iterable(
    glob.glob(str(YOLO_DATASET_DIR / 'images' / 'test' / f'*.{e}'))
    for e in ('jpg', 'jpeg', 'png', 'bmp')))[:8]
print('Running inference on', len(test_imgs), 'test images ...')
det_lists = run_yolov5_inference(
    model, test_imgs, cfg=cfg, conf=0.25, iou=0.45,
)
for path, dets in zip(test_imgs, det_lists):
    print(f'{Path(path).name}: {len(dets)} detections')
    for d in dets:
        print(f'  conf={d.confidence:.2f}  cx={d.cx:.3f} cy={d.cy:.3f} '
              f'w={d.w:.3f} h={d.h:.3f}')
"""))


# ---------------------------------------------------------------------------
# 17. Bounding-box visualization
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 17. Bounding-Box Visualization
"""))
SECTIONS.append(code("""
from PIL import Image
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, path, dets in zip(axes.flat, test_imgs, det_lists):
    im = np.asarray(Image.open(path).convert('RGB'))
    ax.imshow(im)
    boxes = [(d.cx, d.cy, d.w, d.h) for d in dets]
    labels = [f'{d.class_name} {d.confidence:.2f}' for d in dets]
    draw_boxes(ax, boxes, labels=labels, color='red', line_width=1.5)
    ax.set_title(Path(path).name)
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(FIGURES_DIR / '03_detector_predictions.png', dpi=120, bbox_inches='tight')
plt.show()
"""))


# ---------------------------------------------------------------------------
# 18. Narrowband signal extraction
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 18. Narrowband Signal Extraction

The YOLO boxes are converted to frequency / time coordinates and a
bandpass filter extracts the corresponding I/Q clip. The clip is
then padded / trimmed to the classifier's expected length.
"""))
SECTIONS.append(code("""
sample = build_synthetic_wideband(cfg, split='test')[0]
spec = compute_spectrogram(sample.samples, fs=sample.fs,
                           n_fft=cfg.n_fft, hop_length=cfg.hop_length)
# Run detection
tmp_path = FIGURES_DIR / '_tmp.jpg'
Image.fromarray(spectrogram_to_image(spec, image_size=cfg.image_size)).save(tmp_path)
dets = run_yolov5_inference(model, [str(tmp_path)], cfg=cfg, conf=0.25)[0]
print(f'Detections on the test sample: {len(dets)}')

# Convert to DetectedSignal
from src.signal_detection import DetectedSignal
det_objs = []
for i, d in enumerate(dets):
    f_lo, f_hi, t_lo, t_hi = yolo_box_to_freq_time(d.cx, d.cy, d.w, d.h, spec)
    det_objs.append(DetectedSignal(
        signal_id=i+1, time_range=(t_lo, t_hi), frequency_range=(f_lo, f_hi),
        bounding_box_yolo=(d.cx, d.cy, d.w, d.h), confidence=d.confidence,
    ))
clips = extract_narrowband(sample.samples, fs=sample.fs, detections=det_objs,
                           target_len=cfg.clf_seq_len, cfg=cfg)
for c in clips:
    print(f'  signal {c.signal_id}: f=[{c.frequency_range[0]:+.0f}, '
          f'{c.frequency_range[1]:+.0f}] Hz, clip={c.iq.shape}, '
          f'classifier_input={c.classifier_input.shape}')
"""))


# ---------------------------------------------------------------------------
# 19. CNN/ResNet dataset preparation
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 19. CNN / ResNet Dataset Preparation

The narrowband clips are already stored in
`data/narrowband/<class>/*.npy`. We load them as (2, L) arrays.
"""))
SECTIONS.append(code("""
from src.train_classifier import load_narrowband_manifest, stratified_split
X, y, classes = load_narrowband_manifest(str(NARROWBAND_DIR))
print('X shape :', X.shape, 'dtype', X.dtype)
print('y shape :', y.shape, 'classes:', classes)
tr, va, te = stratified_split(y, train=0.7, val=0.15, seed=cfg.seed)
print(f'train={len(tr)}  val={len(va)}  test={len(te)}')
"""))


# ---------------------------------------------------------------------------
# 20. CNN/ResNet training
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 20. CNN / ResNet Training

Train the modulation classifier (default: 1-D ResNet-18, with
optional transfer-learning initialisation from torchvision).
"""))
SECTIONS.append(code("""
import torch
model_clf = build_model(cfg.clf_arch, num_classes=len(classes),
                       in_channels=2, seq_len=cfg.clf_seq_len,
                       use_transfer=cfg.clf_use_transfer)
tcfg = TrainConfig(
    epochs=cfg.clf_epochs,
    batch_size=cfg.clf_batch_size,
    lr=cfg.clf_lr,
    weight_decay=cfg.clf_weight_decay,
    patience=cfg.clf_patience,
    use_augmentation=cfg.clf_use_augmentation,
    use_scheduler=True,
    device=str(get_device()),
)
hist = train_classifier(
    model_clf, X[tr], y[tr], X[va], y[va],
    cfg=tcfg, num_classes=len(classes), verbose=True,
)
torch.save({
    'state_dict': model_clf.state_dict(),
    'arch': cfg.clf_arch,
    'classes': classes,
    'config': cfg.to_dict(),
}, 'models/classifier.pt')
print('Saved classifier to models/classifier.pt')
"""))


# ---------------------------------------------------------------------------
# 21. Classification evaluation
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 21. Classification Evaluation
"""))
SECTIONS.append(code("""
preds, probs = predict_classifier(model_clf, X[te], device=get_device())
metrics = evaluate_classifier(y[te], preds, classes, out_dir=str(METRICS_DIR))
print('Test accuracy :', metrics['accuracy'])
print('Macro F1      :', metrics['f1_macro'])

# Training curves
fig = plot_training_curves(hist, out_path=str(FIGURES_DIR / '04_clf_training.png'))
plt.show()
"""))


# ---------------------------------------------------------------------------
# 22. End-to-end inference
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 22. End-to-End Inference

Combine the detector and the classifier on a fresh wideband capture.
"""))
SECTIONS.append(code("""
sample = build_synthetic_wideband(cfg, split='test')[0]
spec, dets, clips, results = run_end_to_end_inference(
    sample.samples, sample.fs,
    detector_weights=str(yolo_weights),
    classifier_ckpt='models/classifier.pt',
    cfg=cfg,
)
print(f'Detected {len(dets)} signals:')
for r in results:
    print(f\"\"\"  signal {r.signal_id}: f=[{r.frequency_range_hz[0]:+.0f}, {r.frequency_range_hz[1]:+.0f}] Hz
      detector={r.detector_confidence:.2f}  modulation={r.predicted_modulation}
      classifier={r.classifier_confidence:.2f}  topk={r.classifier_topk}\"\"\")
"""))


# ---------------------------------------------------------------------------
# 23. Metrics
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 23. Metrics

Aggregate the detector and classifier metrics that we have already
computed in the previous cells.
"""))
SECTIONS.append(code("""
det_metrics_path = METRICS_DIR / 'detector_metrics.json'
cls_metrics_path = METRICS_DIR / 'classification_metrics.json'
print('Detector (from validation):')
print('  Precision  :', float(val_results.box.mp))
print('  Recall     :', float(val_results.box.mr))
print('  mAP@0.5    :', float(val_results.box.map50))
print('  mAP@0.5:.95:', float(val_results.box.map))
if det_metrics_path.exists():
    d = json.load(open(det_metrics_path))
    print('  IoU (test) :', d.get('mean_iou', 'n/a'))
if cls_metrics_path.exists():
    c = json.load(open(cls_metrics_path))
    print('Classifier (test):')
    print('  Accuracy   :', c['accuracy'])
    print('  Macro F1   :', c['f1_macro'])
"""))


# ---------------------------------------------------------------------------
# 24. Confusion matrix
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 24. Confusion Matrix
"""))
SECTIONS.append(code("""
cm = np.array(metrics['confusion_matrix'])
fig = plot_confusion_matrix(cm, classes, normalize=True,
                            out_path=str(FIGURES_DIR / '05_confusion_matrix.png'))
plt.show()
"""))


# ---------------------------------------------------------------------------
# 25. SNR-wise evaluation
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 25. SNR-Wise Evaluation

Run the detector and classifier on a synthetic dataset that covers
SNRs from -5 dB to 20 dB.
"""))
SECTIONS.append(code("""
sweep_per_snr = 15
det_sweep = snr_sweep_detector(model, cfg=cfg, per_snr=sweep_per_snr)
clf_sweep = snr_sweep_classifier(model_clf, classes, cfg=cfg,
                                 per_snr_per_class=sweep_per_snr)

snr_list = list(det_sweep.keys())
det_p = [det_sweep[s]['precision'] for s in snr_list]
det_r = [det_sweep[s]['recall'] for s in snr_list]
det_iou = [det_sweep[s]['mean_iou'] for s in snr_list]
clf_acc = [clf_sweep[s]['accuracy'] for s in snr_list]
clf_f1 = [clf_sweep[s]['f1_macro'] for s in snr_list]

# Save CSV
import csv
with open(METRICS_DIR / 'snr_sweep.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['SNR_dB', 'Detector_Precision', 'Detector_Recall', 'Detector_IoU',
                'Classifier_Accuracy', 'Classifier_F1'])
    for s, p, r, i, a, f1 in zip(snr_list, det_p, det_r, det_iou, clf_acc, clf_f1):
        w.writerow([s, f'{p:.3f}', f'{r:.3f}', f'{i:.3f}', f'{a:.3f}', f'{f1:.3f}'])

# Plot
plot_metric_vs_snr(snr_list,
                   {'Detector Precision': det_p, 'Detector Recall': det_r},
                   ylabel='Detector metrics',
                   out_path=str(FIGURES_DIR / '06_det_vs_snr.png'))
plt.show()
plot_metric_vs_snr(snr_list,
                   {'Classifier Accuracy': clf_acc, 'Classifier F1': clf_f1},
                   ylabel='Classifier metrics',
                   out_path=str(FIGURES_DIR / '07_clf_vs_snr.png'))
plt.show()
print('Saved SNR table to', METRICS_DIR / 'snr_sweep.csv')
"""))


# ---------------------------------------------------------------------------
# 26. Comparison with paper
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 26. Comparison with Paper

The values reported in the Vagollari et al. (2023) paper are
approximate because the original dataset and code are not public.
We list the headline numbers from the paper next to the numbers we
obtained on our synthetic data.
"""))
SECTIONS.append(code("""
import pandas as pd
paper = {
    'Detector Precision': 0.77,
    'Detector Recall'   : 0.82,
    'Detector mAP@0.5'  : 0.86,
    'Classifier Acc@20dB': 0.95,
    'Classifier Acc@0dB' : 0.60,
}
ours_det = {
    'Detector Precision': float(val_results.box.mp),
    'Detector Recall'   : float(val_results.box.mr),
    'Detector mAP@0.5'  : float(val_results.box.map50),
    'Classifier Acc@20dB': clf_sweep[20.0]['accuracy'] if 20.0 in clf_sweep else float('nan'),
    'Classifier Acc@0dB' : clf_sweep[0.0]['accuracy'] if 0.0 in clf_sweep else float('nan'),
}
df = pd.DataFrame({'Paper (approx.)': paper, 'Our implementation': ours_det})
df['Difference'] = df['Our implementation'] - df['Paper (approx.)']
print(df.round(3).to_markdown())
"""))


# ---------------------------------------------------------------------------
# 27. Limitations
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 27. Limitations

* The original dataset is not released, so the numbers are obtained
  on a **freshly synthesised** dataset. The synthetic generator
  follows the paper's description but cannot be byte-for-byte
  identical.
* The synthetic dataset does not model real-world impairments such
  as IQ imbalance, non-linear amplifiers, or impulsive noise.
* Transfer learning from ImageNet only initialises the deeper
  layers; the first conv layer is re-trained from scratch because
  the input modality (IQ) is different.
* Detector thresholds (`conf=0.25`, `iou=0.45`) are the Ultralytics
  defaults; the paper does not specify which threshold it uses.
"""))


# ---------------------------------------------------------------------------
# 28. Possible improvements
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 28. Possible Improvements

* Replace the synthetic generator with a real captured dataset (e.g.
  over-the-air recordings).
* Try larger YOLO variants (`yolov5m`, `yolov5l`) and longer training.
* Add **per-modulation** YOLO classes to skip the second stage.
* Train the classifier with **mixup** and **SpecAugment**.
* Use a **joint** loss that trains detector + classifier end-to-end.
"""))


# ---------------------------------------------------------------------------
# 29. Final conclusion
# ---------------------------------------------------------------------------
SECTIONS.append(md("""
## 29. Final Conclusion

This notebook reproduced the *spirit* of the Vagollari et al. (2023)
framework: a YOLOv5 detector localises narrowband signals in a
wideband spectrogram, and a 1-D ResNet classifier recognises their
modulation. The full pipeline is implemented in ~1 000 lines of
modular Python and runs end-to-end on Google Colab, Kaggle or a
local GPU. The actual numbers depend strongly on the dataset;
differences with respect to the paper are expected and explicitly
documented in the **Limitations** section.

Happy hacking!
"""))


# ---------------------------------------------------------------------------
# Build & write
# ---------------------------------------------------------------------------
nb = nb_cells(*SECTIONS)
out = Path(__file__).resolve().parents[1] / 'notebooks' / 'wideband_signal_recognition_colab.ipynb'
write_notebook(str(out), nb)
print('Wrote', out)
