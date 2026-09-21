"""Narrowband signal extraction.

Given a wideband signal and a list of `DetectedSignal` objects, this
module extracts the band-limited I/Q clip corresponding to each
detection. The clip can then be fed to the modulation classifier.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal

from .config import Config, get_default_config
from .preprocessing import bandpass_extract
from .signal_detection import DetectedSignal


@dataclass
class ExtractedSignal:
    """A narrowband I/Q clip ready for the classifier."""

    signal_id: int
    iq: np.ndarray                       # complex baseband clip
    time_range: Tuple[float, float]
    frequency_range: Tuple[float, float]
    classifier_input: np.ndarray         # (2, L) real/imag stacked


def extract_narrowband(samples: np.ndarray, fs: float, detections: List[DetectedSignal],
                       target_len: Optional[int] = None,
                       cfg: Optional[Config] = None,
                       pad_factor: float = 1.0) -> List[ExtractedSignal]:
    """Bandpass filter around each detection and produce fixed-length clips.

    Each detection is frequency-shifted to DC and lowpass filtered
    (see `preprocessing.bandpass_extract`), so the returned clip is a
    basebanded narrowband signal centred at 0 Hz.

    Parameters
    ----------
    samples : np.ndarray
        Complex baseband wideband signal.
    fs : float
        Sampling rate (Hz).
    detections : list
        Output of `signal_detection.detect_signals`.
    target_len : int
        Desired clip length. If None, the original signal length is kept.
        Pass `cfg.clf_seq_len` for classifier-ready clips.
    pad_factor : float
        Multiplies the detected bandwidth before bandpass filtering to be
        tolerant to small detection errors.
    """
    if cfg is None:
        cfg = get_default_config()
    out: List[ExtractedSignal] = []
    for det in detections:
        f_lo, f_hi = det.frequency_range
        centre = 0.5 * (f_lo + f_hi)
        bw = max(abs(f_hi - f_lo) * pad_factor, cfg.bandwidth)
        if bw <= 0:
            continue
        clip = bandpass_extract(samples, fs=fs, center_hz=centre, bandwidth_hz=bw)
        if clip.size == 0:
            continue
        # Normalise to unit power
        p = np.mean(np.abs(clip) ** 2)
        if p > 0 and np.isfinite(p):
            clip = clip / np.sqrt(p)
        # Trim / pad to target_len
        if target_len is not None:
            if len(clip) > target_len:
                start = (len(clip) - target_len) // 2
                clip = clip[start:start + target_len]
            elif len(clip) < target_len:
                clip = np.pad(clip, (0, target_len - len(clip)))
        classifier_input = np.stack([clip.real, clip.imag], axis=0).astype(np.float32)
        out.append(ExtractedSignal(
            signal_id=det.signal_id,
            iq=clip.astype(np.complex64),
            time_range=det.time_range,
            frequency_range=det.frequency_range,
            classifier_input=classifier_input,
        ))
    return out


def extract_all(samples: np.ndarray, fs: float, detections: List[DetectedSignal],
                cfg: Optional[Config] = None) -> List[ExtractedSignal]:
    """Convenience wrapper that uses the default classifier sequence length."""
    if cfg is None:
        cfg = get_default_config()
    return extract_narrowband(samples, fs, detections, target_len=cfg.clf_seq_len, cfg=cfg)


# ---------------------------------------------------------------------------
# Optional in-memory dataset object
# ---------------------------------------------------------------------------
@dataclass
class NarrowbandDataset:
    """Container of (X, y) arrays for the modulation classifier."""

    X: np.ndarray        # (N, 2, L) float32
    y: np.ndarray        # (N,) int64
    class_names: List[str]

    def __len__(self) -> int:
        return len(self.X)

    def num_classes(self) -> int:
        return len(self.class_names)
