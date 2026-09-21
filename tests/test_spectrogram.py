"""Unit tests for the spectrogram module."""
import numpy as np
import pytest

from src.spectrogram import compute_spectrogram, spectrogram_to_image, spectrogram_to_grayscale


def test_compute_spectrogram_shape():
    fs = 200_000.0
    t = np.arange(2048) / fs
    sig = np.exp(1j * 2 * np.pi * 30_000 * t)
    spec = compute_spectrogram(sig, fs=fs, n_fft=1024, hop_length=64)
    n_freq, n_time = spec.shape
    assert n_freq == 1024
    assert n_time > 0
    # Frequencies should be ordered (after fftshift)
    assert np.all(np.diff(spec.freq_hz) >= 0)


def test_spectrogram_to_image_shape():
    fs = 200_000.0
    t = np.arange(2048) / fs
    sig = np.exp(1j * 2 * np.pi * 30_000 * t)
    spec = compute_spectrogram(sig, fs=fs)
    img = spectrogram_to_image(spec, image_size=256)
    assert img.shape == (256, 256, 3)
    assert img.dtype == np.uint8


def test_spectrogram_to_grayscale():
    fs = 200_000.0
    sig = np.random.randn(2048) + 1j * np.random.randn(2048)
    spec = compute_spectrogram(sig, fs=fs)
    g = spectrogram_to_grayscale(spec, image_size=128)
    assert g.shape == (128, 128)
    assert g.min() >= 0.0
    assert g.max() <= 1.0


def test_spectrogram_detects_tone():
    fs = 200_000.0
    t = np.arange(8192) / fs
    sig = np.exp(1j * 2 * np.pi * 30_000 * t)
    spec = compute_spectrogram(sig, fs=fs, n_fft=1024, hop_length=64)
    # Find the row with maximum power
    peak_row = int(np.argmax(spec.db.mean(axis=1)))
    peak_freq = spec.freq_hz[peak_row]
    # Closest frequency to 30 kHz should be the peak
    expected = np.argmin(np.abs(spec.freq_hz - 30_000.0))
    assert abs(peak_row - expected) <= 1
