"""Unit tests for the preprocessing module."""
import numpy as np
import pytest

from src.preprocessing import (
    generate_narrowband, generate_wideband, _bits_to_symbols,
    bandpass_extract, add_awgn,
)


def test_bpsk_constellation():
    bits = np.array([0, 1, 0, 1, 1, 0])
    sym = _bits_to_symbols(bits, "BPSK")
    assert sym.shape == (6,)
    # Power is 1
    assert np.isclose(np.mean(np.abs(sym) ** 2), 1.0, atol=1e-6)


def test_qpsk_constellation():
    bits = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    sym = _bits_to_symbols(bits, "QPSK")
    assert sym.shape == (4,)
    assert np.isclose(np.mean(np.abs(sym) ** 2), 1.0, atol=1e-6)


def test_generate_narrowband_shape():
    rng = np.random.default_rng(0)
    sig = generate_narrowband(1024, "QPSK", 20.0, rng)
    assert sig.shape == (1024,)
    assert np.iscomplexobj(sig)
    # Power roughly 1 (within a few dB)
    p = np.mean(np.abs(sig) ** 2)
    assert 0.05 < p < 5.0


def test_noise_classifier():
    rng = np.random.default_rng(1)
    sig = generate_narrowband(512, "NOISE", 20.0, rng)
    assert sig.shape == (512,)
    # NOISE has unit power
    p = np.mean(np.abs(sig) ** 2)
    assert 0.5 < p < 2.0


def test_generate_wideband_signal_count():
    rng = np.random.default_rng(2)
    fs = 200_000.0
    n = int(fs * 0.1)
    s = generate_wideband(num_samples=n, fs=fs, snr_db=20.0,
                          mod_classes=["BPSK", "QPSK", "16QAM"],
                          min_signals=1, max_signals=3,
                          bandwidth=12_000.0, padding=2_000.0, rng=rng)
    assert 1 <= len(s.signals) <= 3
    assert s.samples.shape == (n,)


def test_add_awgn_snr():
    rng = np.random.default_rng(3)
    s = np.ones(8192) + 0j
    noisy = add_awgn(s, snr_db=10.0, rng=rng)
    p_sig = np.mean(np.abs(s) ** 2)
    p_noise = np.mean(np.abs(noisy - s) ** 2)
    snr_emp = 10 * np.log10(p_sig / (p_noise + 1e-12))
    assert 8.0 < snr_emp < 12.0


def test_bandpass_extract():
    fs = 200_000.0
    t = np.arange(8192) / fs
    sig = np.exp(1j * 2 * np.pi * 30_000 * t)
    out = bandpass_extract(sig, fs=fs, center_hz=30_000.0, bandwidth_hz=5_000.0)
    assert out.shape == sig.shape
    p = np.mean(np.abs(out) ** 2)
    assert p > 0.1  # most of the power should be retained
