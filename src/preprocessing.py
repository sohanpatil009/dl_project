"""Preprocessing utilities for the wideband signal recognition pipeline.

This module contains the I/Q signal generators used to synthesise both
the narrowband clips consumed by the modulation classifier and the
wideband captures consumed by the spectrogram + YOLO stage.

The synthetic data closely follows the procedure described in:

  A. Vagollari, M. Hirschbeck, W. Gerstacker,
  "An End-to-End Deep Learning Framework for Wideband Signal Recognition",
  IEEE Access, vol. 11, pp. 52899-52922, 2023.

Because the original dataset is not released, this implementation
*generates its own* dataset with the same parameters. The generator is
fully configurable through `src.config.Config` so it can be replaced
by an external dataset loader when the original data becomes available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal


# ---------------------------------------------------------------------------
# Bit / symbol helpers
# ---------------------------------------------------------------------------
def _bits_to_symbols(bits: np.ndarray, mod: str) -> np.ndarray:
    """Map a stream of bits to complex baseband symbols for the given modulation.

    Constellations use unit average power (normalised at the end).
    """
    mod = mod.upper()
    if mod == "BPSK":
        symbols = np.where(bits == 0, -1.0 + 0j, 1.0 + 0j)
    elif mod == "QPSK":
        # Gray coded 00,01,11,10 -> (-1-1j, -1+1j, 1+1j, 1-1j)/sqrt(2)
        gray = {(0, 0): -1 - 1j, (0, 1): -1 + 1j, (1, 1): 1 + 1j, (1, 0): 1 - 1j}
        pairs = bits.reshape(-1, 2)
        symbols = np.array([gray[(int(b[0]), int(b[1]))] for b in pairs]) / np.sqrt(2)
    elif mod == "8PSK":
        gray = {
            (0, 0, 0): np.exp(1j * 0 * np.pi / 4),
            (0, 0, 1): np.exp(1j * 1 * np.pi / 4),
            (0, 1, 1): np.exp(1j * 2 * np.pi / 4),
            (0, 1, 0): np.exp(1j * 3 * np.pi / 4),
            (1, 1, 0): np.exp(1j * 4 * np.pi / 4),
            (1, 1, 1): np.exp(1j * 5 * np.pi / 4),
            (1, 0, 1): np.exp(1j * 6 * np.pi / 4),
            (1, 0, 0): np.exp(1j * 7 * np.pi / 4),
        }
        triples = bits.reshape(-1, 3)
        symbols = np.array([gray[(int(b[0]), int(b[1]), int(b[2]))] for b in triples])
    elif mod == "16QAM":
        gray4 = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
        quads = bits.reshape(-1, 4)
        I = np.array([gray4[(int(b[0]), int(b[1]))] for b in quads])
        Q = np.array([gray4[(int(b[2]), int(b[3]))] for b in quads])
        symbols = (I + 1j * Q) / np.sqrt(10)
    elif mod == "64QAM":
        gray8 = {
            (0, 0, 0): -7, (0, 0, 1): -5, (0, 1, 1): -3, (0, 1, 0): -1,
            (1, 1, 0): 1, (1, 1, 1): 3, (1, 0, 1): 5, (1, 0, 0): 7,
        }
        sextets = bits.reshape(-1, 6)
        I = np.array([gray8[(int(b[0]), int(b[1]), int(b[2]))] for b in sextets])
        Q = np.array([gray8[(int(b[3]), int(b[4]), int(b[5]))] for b in sextets])
        symbols = (I + 1j * Q) / np.sqrt(42)
    else:
        raise ValueError(f"Unsupported modulation: {mod}")
    return symbols.astype(np.complex128)


def random_bits(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return `n` random bits as int8 values 0/1."""
    return rng.integers(0, 2, size=n, dtype=np.int8)


# ---------------------------------------------------------------------------
# Pulse shaping
# ---------------------------------------------------------------------------
def rrc_taps(num_taps: int, beta: float, samples_per_symbol: int) -> np.ndarray:
    """Root-raised-cosine FIR filter taps."""
    t = (np.arange(num_taps) - (num_taps - 1) / 2) / samples_per_symbol
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 - beta + (4 * beta / np.pi)
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h ** 2))
    return h


def pulse_shape(symbols: np.ndarray, samples_per_symbol: int, beta: float) -> np.ndarray:
    """Upsample symbols with a root-raised-cosine pulse."""
    sps = max(1, int(samples_per_symbol))
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex128)
    upsampled[::sps] = symbols
    num_taps = max(8 * sps + 1, 65)
    if num_taps % 2 == 0:
        num_taps += 1
    taps = rrc_taps(num_taps, beta, sps)
    shaped = np.convolve(upsampled, taps, mode="same")
    return shaped


# ---------------------------------------------------------------------------
# Channel effects
# ---------------------------------------------------------------------------
def add_awgn(signal_in: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex AWGN to achieve the desired SNR (in dB)."""
    power = np.mean(np.abs(signal_in) ** 2)
    if power <= 0 or not np.isfinite(power):
        return signal_in.copy()
    noise_power = power / (10 ** (snr_db / 10.0))
    noise = (rng.normal(0, np.sqrt(noise_power / 2), signal_in.shape)
             + 1j * rng.normal(0, np.sqrt(noise_power / 2), signal_in.shape))
    return signal_in + noise


def add_frequency_offset(signal_in: np.ndarray, fs: float, max_offset_hz: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Apply a small carrier frequency offset (CFO)."""
    if max_offset_hz <= 0:
        return signal_in
    f_off = rng.uniform(-max_offset_hz, max_offset_hz)
    t = np.arange(len(signal_in)) / fs
    return signal_in * np.exp(1j * 2 * np.pi * f_off * t)


def add_phase_offset(signal_in: np.ndarray, rng: np.random.Generator,
                     max_deg: float = 30.0) -> np.ndarray:
    """Apply a random constant phase rotation."""
    if max_deg <= 0:
        return signal_in
    phi = np.deg2rad(rng.uniform(-max_deg, max_deg))
    return signal_in * np.exp(1j * phi)


# ---------------------------------------------------------------------------
# Narrowband signal
# ---------------------------------------------------------------------------
@dataclass
class NarrowbandSignal:
    """A baseband complex narrowband signal.

    Attributes
    ----------
    modulation : str
        Modulation name (BPSK, QPSK, 8PSK, 16QAM, 64QAM or NOISE).
    samples : np.ndarray
        Complex baseband samples (length = num_samples).
    carrier_offset_hz : float
        Frequency shift applied to place the signal inside the wideband
        capture (Hz).
    start_sample : int
        Index of the first sample of the active region (in the wideband
        signal timeline). ``-1`` means the signal spans the whole window.
    """

    modulation: str
    samples: np.ndarray
    carrier_offset_hz: float
    start_sample: int
    snr_db: float = 20.0


def generate_narrowband(num_samples: int, modulation: str, snr_db: float,
                        rng: np.random.Generator, beta: float = 0.35,
                        add_cfo: bool = True, add_phase: bool = True,
                        max_cfo_hz: float = 50.0) -> np.ndarray:
    """Generate a single complex baseband narrowband signal.

    If `modulation` is ``"NOISE"`` only AWGN of unit power is returned.
    """
    if modulation.upper() == "NOISE":
        s = (rng.normal(0, 1 / np.sqrt(2), num_samples)
             + 1j * rng.normal(0, 1 / np.sqrt(2), num_samples))
        return s.astype(np.complex128)
    if modulation.upper() == "NO_SIGNAL":
        return np.zeros(num_samples, dtype=np.complex128)

    sps = 8  # samples per symbol; high enough for clean pulse shaping
    bits_per_symbol = {
        "BPSK": 1, "QPSK": 2, "8PSK": 3, "16QAM": 4, "64QAM": 6,
    }[modulation.upper()]
    n_symbols = int(np.ceil(num_samples / sps)) + 32
    bits = random_bits(n_symbols * bits_per_symbol, rng)
    symbols = _bits_to_symbols(bits, modulation)
    sig = pulse_shape(symbols, sps, beta)
    # Trim / pad to num_samples
    if len(sig) > num_samples:
        start = (len(sig) - num_samples) // 2
        sig = sig[start:start + num_samples]
    elif len(sig) < num_samples:
        sig = np.pad(sig, (0, num_samples - len(sig)))
    sig = sig / np.sqrt(np.mean(np.abs(sig) ** 2) + 1e-12)
    if add_cfo:
        # Internal CFO in baseband - very small, mostly for realism
        sig = add_frequency_offset(sig, fs=1.0, max_offset_hz=max_cfo_hz / 1e6, rng=rng)
    if add_phase:
        sig = add_phase_offset(sig, rng, max_deg=15.0)
    sig = add_awgn(sig, snr_db, rng)
    return sig.astype(np.complex128)


# ---------------------------------------------------------------------------
# Wideband mixing
# ---------------------------------------------------------------------------
@dataclass
class WidebandSample:
    """Container for a synthesised wideband capture."""

    samples: np.ndarray                              # complex baseband I/Q
    signals: List[NarrowbandSignal] = field(default_factory=list)
    fs: float = 200_000.0
    snr_db: float = 20.0


def generate_wideband(num_samples: int, fs: float, snr_db: float,
                      mod_classes: List[str], min_signals: int, max_signals: int,
                      bandwidth: float, padding: float, rng: np.random.Generator,
                      beta: float = 0.35,
                      allow_noise_class: bool = True) -> WidebandSample:
    """Generate one wideband capture containing 1..max_signals narrowband sigs.

    The active modulation classes exclude ``NO_SIGNAL``. ``NOISE`` is kept
    as a real emission (band-limited white noise) only if
    ``allow_noise_class`` is True.
    """
    sig_pool = [c for c in mod_classes if c != "NO_SIGNAL"]
    n_signals = int(rng.integers(min_signals, max_signals + 1))
    available_bw = fs / 2 - (bandwidth + padding)
    if available_bw <= 0:
        raise ValueError("Wideband bandwidth too small for the requested narrowband width.")
    # Random carrier offsets (avoid overlap using minimum spacing)
    min_spacing = bandwidth + padding
    if n_signals == 1:
        offsets = [float(rng.uniform(-available_bw / 2, available_bw / 2))]
    else:
        # Spread signals uniformly
        offsets = np.linspace(-available_bw / 2, available_bw / 2, n_signals)
        rng.shuffle(offsets)
    narrowband_list: List[NarrowbandSignal] = []
    wideband = np.zeros(num_samples, dtype=np.complex128)
    for k in range(n_signals):
        mod = sig_pool[int(rng.integers(0, len(sig_pool)))]
        if mod == "NOISE" and not allow_noise_class:
            mod = "BPSK"
        nbs = generate_narrowband(num_samples, mod, snr_db, rng, beta=beta)
        narrowband_list.append(
            NarrowbandSignal(
                modulation=mod,
                samples=nbs,
                carrier_offset_hz=float(offsets[k]),
                start_sample=0,
                snr_db=snr_db,
            )
        )
        t = np.arange(num_samples) / fs
        wideband += nbs * np.exp(1j * 2 * np.pi * offsets[k] * t)
    # Add global AWGN to reach the *target* SNR after combining
    wideband = add_awgn(wideband, snr_db, rng)
    return WidebandSample(samples=wideband, signals=narrowband_list, fs=fs, snr_db=snr_db)


# ---------------------------------------------------------------------------
# Narrowband clip extraction (used by the classifier)
# ---------------------------------------------------------------------------
def bandpass_extract(samples: np.ndarray, fs: float, center_hz: float,
                     bandwidth_hz: float) -> np.ndarray:
    """Extract a band around `center_hz` of width `bandwidth_hz`.

    The wideband captures in this project are *complex baseband* signals
    whose carriers can sit at negative frequencies (e.g. -43 kHz). A
    classic real-valued bandpass filter cannot represent those, so for
    complex inputs we frequency-shift the band of interest to DC and
    apply a lowpass filter instead. The returned clip is therefore a
    basebanded narrowband signal at 0 Hz. Real-valued inputs keep the
    legacy bandpass behaviour.
    """
    if bandwidth_hz <= 0:
        return np.array([], dtype=np.complex128)
    if np.iscomplexobj(samples):
        # Shift band centre to DC, then lowpass at half bandwidth.
        n = len(samples)
        t = np.arange(n) / fs
        shifted = samples * np.exp(-1j * 2 * np.pi * center_hz * t)
        cutoff = min(bandwidth_hz / 2, fs / 2 - 1.0)
        if cutoff <= 0:
            return np.array([], dtype=np.complex128)
        nyq = fs / 2
        sos = scipy_signal.butter(6, cutoff / nyq, btype="low", output="sos")
        filtered = scipy_signal.sosfiltfilt(sos, shifted)
        return filtered
    half = bandwidth_hz / 2
    low = max(center_hz - half, 1.0)
    high = min(center_hz + half, fs / 2 - 1.0)
    if high <= low:
        return np.array([], dtype=np.complex128)
    # scipy uses normalised frequencies 0..1 (1 = Nyquist)
    nyq = fs / 2
    sos = scipy_signal.butter(6, [low / nyq, high / nyq], btype="band", output="sos")
    filtered = scipy_signal.sosfiltfilt(sos, samples)
    return filtered
