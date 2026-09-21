# Results

Experiment outputs are written here by the notebooks and by
`scripts/evaluate.py` / `scripts/inference.py`. Nothing in this tree is
committed to git except this README (see `.gitignore`).

```
results/
├── figures/       # publication-quality PNGs (spectrograms, curves, confusion matrices, ...)
├── metrics/       # config.yaml, *_metrics.json, snr_sweep.csv, training histories
└── predictions/   # end_to_end.json and per-image detections
```

To reproduce the reference run:

```bash
python scripts/prepare_dataset.py
python scripts/train_detector.py --data-yaml data/yolo_dataset/data.yaml
python scripts/train_classifier.py --data data/narrowband
python scripts/evaluate.py --detector-weights models/runs/detector/weights/best.pt \
                           --classifier-ckpt models/classifier.pt --snr-sweep
```
