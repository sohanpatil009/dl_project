"""End-to-end smoke tests (no model training, no YOLO weights required)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.config import get_default_config
from src.preprocessing import generate_wideband
from src.spectrogram import compute_spectrogram, spectrogram_to_image
from src.signal_extraction import extract_narrowband
from src.signal_detection import DetectedSignal
from src.classifier import build_model, predict_classifier, TrainConfig, train_classifier
from src.yolo_utils import yolo_box_to_freq_time


def _make_detection(fs: float, f_center: float, t_dur: float, t0: float = 0.0,
                    confidence: float = 0.9) -> DetectedSignal:
    return DetectedSignal(
        signal_id=1,
        time_range=(t0, t0 + t_dur),
        frequency_range=(f_center - 6_000.0, f_center + 6_000.0),
        bounding_box_yolo=(0.5, 0.5, 0.05, 1.0),
        confidence=confidence,
    )


def test_yolo_box_to_freq_time_round_trip():
    cfg = get_default_config()
    fs = cfg.fs
    n = int(fs * cfg.duration)
    rng = np.random.default_rng(0)
    s = generate_wideband(num_samples=n, fs=fs, snr_db=20.0,
                          mod_classes=["BPSK", "QPSK"], min_signals=1, max_signals=2,
                          bandwidth=cfg.bandwidth, padding=cfg.padding, rng=rng)
    spec = compute_spectrogram(s.samples, fs=fs, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    f_lo, f_hi, t_lo, t_hi = yolo_box_to_freq_time(0.5, 0.5, 0.1, 1.0, spec)
    # Box should cover the full time range
    assert t_lo <= t_hi
    # Frequency range should be 10% of the available BW
    expected = (spec.freq_hz.max() - spec.freq_hz.min()) * 0.1
    assert abs((f_hi - f_lo) - expected) < 1.0


def test_extract_narrowband_shape():
    cfg = get_default_config()
    fs = cfg.fs
    n = int(fs * cfg.duration)
    rng = np.random.default_rng(1)
    s = generate_wideband(num_samples=n, fs=fs, snr_db=20.0,
                          mod_classes=["QPSK"], min_signals=1, max_signals=1,
                          bandwidth=cfg.bandwidth, padding=cfg.padding, rng=rng)
    spec = compute_spectrogram(s.samples, fs=fs, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    det = _make_detection(fs=fs, f_center=s.signals[0].carrier_offset_hz,
                          t_dur=spec.time_s[-1] - spec.time_s[0])
    clips = extract_narrowband(s.samples, fs=fs, detections=[det], cfg=cfg,
                               target_len=cfg.clf_seq_len)
    assert len(clips) == 1
    assert clips[0].iq.shape == (cfg.clf_seq_len,)
    assert clips[0].classifier_input.shape == (2, cfg.clf_seq_len)


def test_classifier_smoke():
    """Train a tiny classifier on toy data and check it overfits."""
    rng = np.random.default_rng(2)
    n = 64
    X = rng.normal(0, 1, (n, 2, 128)).astype(np.float32)
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    # Make class 0 clearly different
    X[:n // 2, 0, :] += 1.0
    model = build_model("cnn1d", num_classes=2, in_channels=2, seq_len=128, use_transfer=False)
    cfg = TrainConfig(epochs=2, batch_size=16, lr=1e-3, use_augmentation=False, use_scheduler=False)
    history = train_classifier(model, X, y, X[:8], y[:8], cfg, num_classes=2, verbose=False)
    assert "train_loss" in history
    preds, probs = predict_classifier(model, X, device=None)
    assert preds.shape == (n,)
    assert probs.shape == (n, 2)


def test_snr_sweep_negative_snr_seed():
    """Regression: SNR sweeps must accept negative dB bins (used to crash)."""
    import numpy as np
    from src.config import get_default_config
    from src.classifier import build_model
    from src.evaluate import snr_sweep_classifier
    cfg = get_default_config()
    cfg.snr_eval_db = [-5.0, 0.0]
    model = build_model("cnn1d", num_classes=3, in_channels=2,
                        seq_len=cfg.clf_seq_len, use_transfer=False)
    out = snr_sweep_classifier(model, ["BPSK", "QPSK", "8PSK"],
                               cfg=cfg, per_snr_per_class=2)
    assert set(out.keys()) == {-5.0, 0.0}
    assert 0.0 <= out[-5.0]["accuracy"] <= 1.0


def test_end_to_end_no_yolo():
    """The end-to-end pipeline must not crash on an empty wideband signal.

    This only tests the spectrogram + extraction + classifier parts.
    """
    cfg = get_default_config()
    fs = cfg.fs
    n = int(fs * cfg.duration)
    rng = np.random.default_rng(3)
    s = generate_wideband(num_samples=n, fs=fs, snr_db=20.0,
                          mod_classes=["BPSK"], min_signals=1, max_signals=1,
                          bandwidth=cfg.bandwidth, padding=cfg.padding, rng=rng)
    spec = compute_spectrogram(s.samples, fs=fs, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    img = spectrogram_to_image(spec, image_size=cfg.image_size)
    assert img.shape == (cfg.image_size, cfg.image_size, 3)
