"""Full pipeline smoke test: classifier training + mock end-to-end."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.config import get_default_config, set_seed, get_device
from src.train_classifier import load_narrowband_manifest, stratified_split
from src.classifier import build_model, train_classifier, TrainConfig, predict_classifier
from src.evaluate import evaluate_classifier
from src.preprocessing import generate_wideband
from src.spectrogram import compute_spectrogram
from src.signal_detection import DetectedSignal
from src.signal_extraction import extract_narrowband

set_seed(0)
cfg = get_default_config()

X, y, classes = load_narrowband_manifest('data_smoke/narrowband')
print('loaded:', X.shape, classes)
tr, va, te = stratified_split(y, train=0.7, val=0.15, seed=cfg.seed)
model = build_model('cnn1d', num_classes=len(classes), in_channels=2,
                    seq_len=cfg.clf_seq_len, use_transfer=False)
tcfg = TrainConfig(epochs=3, batch_size=16, lr=1e-3, patience=3,
                   use_augmentation=True, use_scheduler=False,
                   device=str(get_device()))
hist = train_classifier(model, X[tr], y[tr], X[va], y[va],
                        cfg=tcfg, num_classes=len(classes), verbose=True)
preds, probs = predict_classifier(model, X[te], device=get_device())
m = evaluate_classifier(y[te], preds, classes)
print('smoke test acc: %.3f macroF1: %.3f' % (m['accuracy'], m['f1_macro']))

# Mock end-to-end: synthetic wideband -> GT-derived boxes -> extract -> classify
rng = np.random.default_rng(0)
from src.preprocessing import generate_wideband as gw
s = gw(num_samples=int(cfg.fs * cfg.duration), fs=cfg.fs, snr_db=20.0,
       mod_classes=cfg.mod_classes, min_signals=2, max_signals=2,
       bandwidth=cfg.bandwidth, padding=cfg.padding, rng=rng)
spec = compute_spectrogram(s.samples, fs=s.fs, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
dets = []
for i, nbs in enumerate(s.signals):
    f_lo = float(spec.freq_hz.min()); f_hi = float(spec.freq_hz.max())
    t_lo = float(spec.time_s.min()); t_hi = float(spec.time_s.max())
    cx = (nbs.carrier_offset_hz - f_lo) / (f_hi - f_lo)
    dets.append(DetectedSignal(signal_id=i + 1, time_range=(t_lo, t_hi),
                               frequency_range=(nbs.carrier_offset_hz - 6000,
                                                nbs.carrier_offset_hz + 6000),
                               bounding_box_yolo=(cx, 0.5, 0.06, 1.0),
                               confidence=0.99))
clips = extract_narrowband(s.samples, fs=s.fs, detections=dets,
                           target_len=cfg.clf_seq_len, cfg=cfg)
print('extracted %d clips, shapes %s' % (len(clips), [c.classifier_input.shape for c in clips]))
Xq = np.stack([c.classifier_input for c in clips], axis=0)
p, pr = predict_classifier(model, Xq, device=get_device())
for i, c in enumerate(clips):
    print('  gt=%s pred=%s conf=%.2f' % (s.signals[i].modulation, classes[int(p[i])], float(pr[i].max())))
print('SMOKE OK')
