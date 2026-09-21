"""Spectrogram generation utilities.

This module converts a complex baseband wideband signal into a 2-D
intensity image that is suitable for both human inspection and the
YOLOv5 detector. The pipeline follows the description in Section
III-B of the Vagollari et al. paper:

1. STFT with configurable FFT size / hop length / window
2. Convert to dB, clip to a dynamic range
3. Resize to the YOLO input resolution
4. Normalise to 0..255 uint8
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal
from PIL import Image


# ---------------------------------------------------------------------------
# Spectrogram container
# ---------------------------------------------------------------------------
@dataclass
class Spectrogram:
    """A 2-D power spectrogram expressed in decibels.

    Attributes
    ----------
    db : np.ndarray
        ``(n_freq, n_time)`` array of dB values.
    freq_hz : np.ndarray
        Frequency axis (Hz), length ``n_freq``.
    time_s : np.ndarray
        Time axis (s), length ``n_time``.
    fs : float
        Sampling rate (Hz).
    n_fft : int
        FFT size used.
    hop_length : int
        Hop length used.
    """

    db: np.ndarray
    freq_hz: np.ndarray
    time_s: np.ndarray
    fs: float
    n_fft: int
    hop_length: int

    @property
    def shape(self) -> Tuple[int, int]:
        return self.db.shape


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def compute_spectrogram(samples: np.ndarray, fs: float, n_fft: int = 1024,
                        hop_length: int = 64, window: str = "hann",
                        db_floor: float = -120.0) -> Spectrogram:
    """Compute a magnitude spectrogram in dB.

    Parameters
    ----------
    samples : np.ndarray
        Complex baseband signal.
    fs : float
        Sampling rate (Hz).
    n_fft : int
        FFT size.
    hop_length : int
        Hop length (must be ``< n_fft``).
    window : str
        Any scipy/numpy window name (default ``"hann"``).
    db_floor : float
        Floor for the dB scale (anything below is clipped).
    """
    if hop_length >= n_fft:
        raise ValueError("`hop_length` must be strictly smaller than `n_fft`.")
    if samples.size == 0:
        raise ValueError("Cannot compute spectrogram of an empty signal.")
    if np.iscomplexobj(samples):
        samples = samples.astype(np.complex128)
    else:
        samples = samples.astype(np.float64) + 0j

    _, _, Z = scipy_signal.stft(
        samples, fs=fs, window=window, nperseg=n_fft, noverlap=n_fft - hop_length,
        nfft=n_fft, return_onesided=False, boundary=None, padded=False,
    )
    # FFT shift so that 0 Hz is in the middle of the image (matches paper figures)
    Z = np.fft.fftshift(Z, axes=0)
    power = np.abs(Z) ** 2
    # Avoid log(0)
    power = np.maximum(power, 1e-12)
    db = 10.0 * np.log10(power)
    db = np.maximum(db, db_floor)

    n_freq, n_time = Z.shape
    freq_hz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / fs))
    time_s = np.arange(n_time) * hop_length / fs
    return Spectrogram(db=db, freq_hz=freq_hz, time_s=time_s, fs=fs,
                       n_fft=n_fft, hop_length=hop_length)


def spectrogram_to_image(spec: Spectrogram, image_size: int = 1024,
                         cmap_range_db: Optional[Tuple[float, float]] = None,
                         colormap: str = "viridis") -> np.ndarray:
    """Convert a Spectrogram to a normalised RGB uint8 image.

    The image is built by:
    * clipping the dB range (default = full range present in the spec)
    * mapping the dB values to ``[0, 255]``
    * resizing to ``(image_size, image_size)`` using bilinear interpolation
    * replicating the single channel to 3 channels (YOLOv5 expects RGB)
    """
    arr = spec.db
    if cmap_range_db is None:
        vmin, vmax = float(arr.min()), float(arr.max())
    else:
        vmin, vmax = cmap_range_db
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    img = (norm * 255.0).astype(np.uint8)
    # Resize with PIL (antialiased)
    pil = Image.fromarray(img, mode="L").resize((image_size, image_size),
                                                resample=Image.BILINEAR)
    img = np.array(pil)
    img = np.stack([img, img, img], axis=-1)  # 3-channel RGB
    return img


def spectrogram_to_grayscale(spec: Spectrogram, image_size: int = 1024) -> np.ndarray:
    """Return a (H, W) float32 image normalised to [0, 1]."""
    arr = spec.db
    vmin, vmax = float(arr.min()), float(arr.max())
    norm = (arr - vmin) / (vmax - vmin + 1e-12)
    pil = Image.fromarray((norm * 255).astype(np.uint8), mode="L").resize(
        (image_size, image_size), resample=Image.BILINEAR
    )
    return np.asarray(pil, dtype=np.float32) / 255.0
