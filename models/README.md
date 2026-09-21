# Models directory

Trained checkpoints are stored here. They are **not committed** to git.

```
models/
├── classifier.pt                  # modulation classifier checkpoint
├── runs/
│   ├── detector/weights/best.pt   # YOLOv5 detector checkpoint
│   └── detector_val/              # validation outputs
└── yolov5/                        # (optional) legacy YOLOv5 checkout
```

## Expected checkpoint formats

**Classifier** (`classifier.pt`, produced by `scripts/train_classifier.py`):

```python
{
  "state_dict": ...,   # model weights
  "arch": "resnet18",  # cnn1d | resnet18 | resnet34
  "classes": [...],    # class names
  "test_acc": float,
  "config": {...},
}
```

**Detector** (`best.pt`, produced by Ultralytics YOLO training via
`scripts/train_detector.py`): standard Ultralytics YOLOv5 checkpoint,
loadable with `ultralytics.YOLO(path)`.
