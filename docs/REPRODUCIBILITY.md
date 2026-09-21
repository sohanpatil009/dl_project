# Reproducibility notes

This file records everything needed to reproduce an experiment bit-for-bit
(except GPU nondeterminism, which is documented below).

## Seeds

* Single global seed: `seed: 42` in `configs/default.yaml` (`src/config.py:DEFAULT_SEED`).
* `src.config.set_seed()` seeds `random`, `numpy`, `torch` (CPU + all CUDA
  devices) and sets `PYTHONHASHSEED`.
* Per-split RNG streams are derived deterministically:
  train → `seed`, val → `seed+1`, test → `seed+2`, SNR sweep → `seed+1000+i`.

## Versions (reference environment)

Record your own with `pip freeze` / `python -m torch.utils.collect_env`:

* Python 3.10/3.11, torch ≥ 2.0, torchvision ≥ 0.15
* numpy ≥ 1.23,<2.0, scipy ≥ 1.10, scikit-learn ≥ 1.2
* ultralytics ≥ 8.0.0 (YOLOv5 runtime), Pillow ≥ 9.0, matplotlib ≥ 3.6
* See `requirements.txt` / `environment.yml` for the full pinned set.

## GPU nondeterminism

cuDNN convolutions are nondeterministic by default. We do **not** force
`torch.use_deterministic_algorithms(True)` because it slows training ~2-3x
and breaks some Ultralytics ops. Expect ±1-2 % metric jitter across runs on
different GPUs; trends (accuracy ↑ with SNR) must be stable.

## Config & artefacts per run

Every training run saves:

* `results/metrics/config.yaml` — the exact `Config` used
* `models/classifier.pt` — `{state_dict, arch, classes, test_acc, config}`
* `models/runs/detector/weights/best.pt` — Ultralytics checkpoint
* `results/metrics/*_metrics.json`, `snr_sweep.csv`
* `results/figures/*.png`

## Dataset version

The synthetic generator has no external version; the effective version is
`git rev-parse HEAD` + the config file. If you change any generator parameter,
treat it as a new dataset version and note it in your report.
