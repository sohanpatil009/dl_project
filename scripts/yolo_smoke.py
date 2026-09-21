"""Best-effort YOLO smoke: 1 epoch on 6 images @320px on CPU.

Proves the detector training + inference path end-to-end. NOT a real
training run (too little data / 1 epoch) — metrics are meaningless here
by design; see the Colab/Kaggle notebooks for the full run.
"""
import sys, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_default_config
from src.yolo_utils import train_yolov5, load_yolov5, run_yolov5_inference
from src.dataset_utils import write_yolo_yaml

cfg = get_default_config()
src = Path("data/yolo_dataset")
tmp = Path("data_yolo_smoke")
if tmp.exists():
    shutil.rmtree(tmp)
for sub in ("train", "val"):
    (tmp / "images" / sub).mkdir(parents=True, exist_ok=True)
    (tmp / "labels" / sub).mkdir(parents=True, exist_ok=True)
    imgs = sorted((src / "images" / sub).glob("*.jpg"))[: (6 if sub == "train" else 2)]
    for im in imgs:
        shutil.copy(im, tmp / "images" / sub / im.name)
        lb = src / "labels" / sub / (im.stem + ".txt")
        shutil.copy(lb, tmp / "labels" / sub / lb.name)
write_yolo_yaml(str(tmp / "data.yaml"), str(tmp), num_classes=1,
                class_names=["signal"])

print("training YOLOv5s: 1 epoch, 6 imgs, 320px, CPU ...")
best = train_yolov5(data_yaml=str(tmp / "data.yaml"), cfg=cfg,
                    epochs=1, batch=2, img_size=320,
                    weights="yolov5s.pt",
                    project="models/runs", name="smoke", device="cpu")
print("best:", best)

model = load_yolov5(str(best))
test_img = sorted((src / "images" / "test").glob("*.jpg"))[0]
dets = run_yolov5_inference(model, [str(test_img)], cfg=cfg,
                            device="cpu")[0]
print("detections on %s: %d" % (test_img.name, len(dets)))
for d in dets:
    print("  conf=%.2f cx=%.3f cy=%.3f w=%.3f h=%.3f" % (
        d.confidence, d.cx, d.cy, d.w, d.h))
shutil.rmtree(tmp, ignore_errors=True)
print("YOLO SMOKE OK")
