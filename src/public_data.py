"""Public internet data support (RadioML 2016.10A).

The paper's original dataset is not public. To train/evaluate on *real*
(recorded-with-impairments) data instead of our pure synthetic generator,
this module wires in the closest legitimate public dataset:

* **RadioML 2016.10A** (O'Shea & West, DeepSig, GNU Radio Conference 2016):
  11 modulations x 20 SNRs (-20...+18 dB, 2 dB steps) x 1000 clips,
  each clip a (2, 128) float32 I/Q array. GNU-Radio synthesis with
  multipath fading, carrier/frequency offsets and realistic impairments.

Coverage of *both* pipeline stages:

* **Narrowband / classifier stage** -- real clips are used directly
  (they are already 2x128, exactly `cfg.clf_seq_len`).
* **Wideband / detector stage** -- real clips are tiled in time to fill
  each narrowband emission, frequency-shifted to random carriers, mixed
  (1-4 signals) + AWGN, then STFT -> spectrogram **images**
  (.jpg/.png/.bmp, configurable) + YOLO boxes. The mixing step is an
  informed adaptation (documented, not claimed as original data).

Modulation mapping to the paper's 7 classes::

    RadioML        -> paper
    BPSK           -> BPSK
    QPSK           -> QPSK
    8PSK           -> 8PSK
    QAM16          -> 16QAM
    QAM64          -> 64QAM
    CPFSK/GFSK/PAM4/AM-DSB/AM-SSB/WBFM -> extra (opt-in, NOT paper classes)
    NOISE / NO_SIGNAL                  -> synthesised locally (not in RadioML)

Sources (2026, verified):

* Colab / local: Zenodo mirror ``10.5281/zenodo.18397070``
  (bit-identical, pickle-protocol-4, ~213 MB tar.bz2).
  Direct file URL: ``https://zenodo.org/api/records/18397070/files/RML2016.10a.tar.bz2/content``
* Kaggle: attach any ``RML2016.10a_dict.pkl`` mirror via **+ Add Data**
  (e.g. ``zaslee/rml2016-10a``, ``nolasthitnotomorrow/radioml2016-deepsigcom``);
  it appears under ``/kaggle/input/...`` and is auto-discovered.
* License: CC BY-NC-SA 4.0 (non-commercial research OK, cite DeepSig).
"""
from __future__ import annotations

import os
import pickle
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .preprocessing import NarrowbandSignal, WidebandSample, add_awgn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ZENODO_API_URL = "https://zenodo.org/api/records/18397070/files/RML2016.10a.tar.bz2/content"
ZENODO_DOI = "10.5281/zenodo.18397070"

RADIOML_MODS: List[str] = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
    "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM",
]
RADIOML_SNRS: List[int] = list(range(-20, 20, 2))  # -20..18

#: RadioML name -> paper name (only the 5 overlapping digital modulations)
OVERLAP_MAP: Dict[str, str] = {
    "BPSK": "BPSK",
    "QPSK": "QPSK",
    "8PSK": "8PSK",
    "QAM16": "16QAM",
    "QAM64": "64QAM",
}
#: Everything else present in RadioML but NOT a paper class
EXTRA_MODS: List[str] = [m for m in RADIOML_MODS if m not in OVERLAP_MAP]


# ---------------------------------------------------------------------------
# Download / discovery
# ---------------------------------------------------------------------------
def _progress_hook(blocks: int, block_size: int, total: int) -> None:
    done = blocks * block_size
    pct = 100.0 * done / max(1, total)
    print(f"\r[public_data] download {done/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)",
          end="", flush=True)


def download_radioml(dest_dir: str,
                     url: str = ZENODO_API_URL) -> str:
    """Download + extract the RadioML 2016.10A mirror. Returns the .pkl path.

    Idempotent: skips download if a ``RML2016.10a_dict.pkl`` already exists
    in `dest_dir`.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for cand in (dest / "RML2016.10a_dict.pkl", dest / "RML2016.10A_dict.pkl"):
        if cand.is_file():
            print(f"[public_data] already present: {cand}")
            return str(cand)
    archive = dest / "RML2016.10a.tar.bz2"
    if not archive.is_file():
        print(f"[public_data] downloading RadioML 2016.10A mirror (~213 MB)\n  from {url}")
        print(f"  DOI: {ZENODO_DOI} | license CC BY-NC-SA 4.0")
        urllib.request.urlretrieve(url, str(archive), reporthook=_progress_hook)
        print("\ndownload complete.")
    else:
        print(f"[public_data] archive already present: {archive}")
    # Extract: the tarball contains the .pkl (name may vary in case)
    print("[public_data] extracting ...")
    with tarfile.open(archive, "r:*") as tf:
        members = [m for m in tf.getmembers() if m.name.lower().endswith(".pkl")]
        if not members:
            tf.extractall(dest)
        else:
            for m in members:
                tf.extract(m, dest)
                # Normalise to the canonical name
                src = dest / m.name
                dst = dest / "RML2016.10a_dict.pkl"
                if src.resolve() != dst.resolve():
                    if dst.exists():
                        dst.unlink()
                    src.rename(dst)
    pkl = dest / "RML2016.10a_dict.pkl"
    if not pkl.is_file():
        # Fall back: search whatever .pkl appeared
        found = sorted(dest.glob("*.pkl"))
        if not found:
            raise FileNotFoundError(f"Extraction produced no .pkl in {dest}")
        return str(found[0])
    print(f"[public_data] ready: {pkl} ({pkl.stat().st_size/1e6:.1f} MB)")
    return str(pkl)


def find_kaggle_radioml(search_roots: Optional[List[str]] = None) -> Optional[str]:
    """Search Kaggle input dirs for an attached RadioML .pkl. Returns path or None."""
    roots = search_roots or ["/kaggle/input"]
    for root in roots:
        r = Path(root)
        if not r.is_dir():
            continue
        for cand in r.rglob("*.pkl"):
            if "rml2016" in cand.name.lower() or "radioml" in cand.name.lower():
                return str(cand)
    return None


def load_radioml_pkl(pkl_path: str) -> Dict[Tuple[str, int], np.ndarray]:
    """Load the RadioML dict: keys (MOD, snr) -> (1000, 2, 128) float32."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(f"{pkl_path} does not look like a RadioML dict.")
    return data


def radioml_summary(data: Dict[Tuple[str, int], np.ndarray]) -> Dict[str, list]:
    """Return {mod: sorted snrs} present in the dict (for logging)."""
    out: Dict[str, list] = {}
    for (mod, snr) in data.keys():
        out.setdefault(str(mod), []).append(int(snr))
    return {m: sorted(v) for m, v in sorted(out.items())}


# ---------------------------------------------------------------------------
# Narrowband clips (classifier stage uses real clips directly)
# ---------------------------------------------------------------------------
def radioml_to_clips(data: Dict[Tuple[str, int], np.ndarray],
                     snr_db: float,
                     per_class: int,
                     rng: np.random.Generator,
                     include_extra: bool = False,
                     nearest_snr: bool = True) -> Dict[str, List[np.ndarray]]:
    """Convert a RadioML dict to {paper_class: [complex128 (128,)]}.

    * Only the 5 overlapping modulations are kept by default.
    * `include_extra=True` additionally keeps CPFSK/GFSK/PAM4/AM-*/WBFM
      under their ORIGINAL names (documented as non-paper classes).
    * NOISE / NO_SIGNAL are synthesised upstream (not in RadioML).
    * If the exact `snr_db` is absent, the nearest available SNR is used.
    """
    # Group available SNRs per mod
    avail: Dict[str, List[int]] = {}
    for (mod, snr) in data.keys():
        avail.setdefault(str(mod), []).append(int(snr))
    wanted = list(OVERLAP_MAP.keys()) + (EXTRA_MODS if include_extra else [])
    out: Dict[str, List[np.ndarray]] = {}
    for rml_mod in wanted:
        if rml_mod not in avail:
            continue
        snrs = sorted(avail[rml_mod])
        if int(snr_db) in snrs:
            use_snr = int(snr_db)
        elif nearest_snr:
            use_snr = min(snrs, key=lambda s: abs(s - snr_db))
        else:
            continue
        arr = np.asarray(data[(rml_mod, use_snr)])  # (1000, 2, 128)
        n = min(per_class, len(arr))
        idx = rng.choice(len(arr), size=n, replace=False)
        paper_name = OVERLAP_MAP.get(rml_mod, rml_mod)
        clips = []
        for i in idx:
            iq = arr[int(i)].astype(np.float64)  # (2, 128)
            c = (iq[0] + 1j * iq[1]).astype(np.complex128)
            clips.append(c)
        out[paper_name] = clips
    return out


# ---------------------------------------------------------------------------
# Wideband mixing from real clips (detector stage)
# ---------------------------------------------------------------------------
def tile_clip_to_length(clip128: np.ndarray, num_samples: int,
                        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Tile a 128-sample real clip to `num_samples` (with random offset).

    Tiling is an informed adaptation: RadioML clips are short, so repetition
    is the only way to fill a 0.5 s wideband emission while keeping the real
    modulation + impairments. Boundary discontinuities are accepted and
    documented.
    """
    clip128 = np.asarray(clip128).ravel()
    if len(clip128) == 0:
        raise ValueError("empty clip")
    start = 0
    if rng is not None:
        start = int(rng.integers(0, len(clip128)))
        clip128 = np.concatenate([clip128[start:], clip128[:start]])
    reps = int(np.ceil(num_samples / len(clip128)))
    tiled = np.tile(clip128, reps)[:num_samples]
    return tiled.astype(np.complex128)


def generate_wideband_from_real(num_samples: int, fs: float, snr_db: float,
                                real_pool: Dict[str, List[np.ndarray]],
                                mod_classes: List[str],
                                min_signals: int, max_signals: int,
                                bandwidth: float, padding: float,
                                rng: np.random.Generator) -> WidebandSample:
    """Mix 1..max_signals *real* narrowband emissions into one capture.

    `real_pool` maps modulation name -> list of 128-sample complex clips
    (as returned by `radioml_to_clips`). Classes absent from the pool
    (e.g. NOISE/NO_SIGNAL) fall back to local synthesis for that emission.
    """
    from .preprocessing import generate_narrowband  # local import: avoid cycle

    usable = [c for c in mod_classes if c != "NO_SIGNAL"]
    n_signals = int(rng.integers(min_signals, max_signals + 1))
    available_bw = fs / 2 - (bandwidth + padding)
    if available_bw <= 0:
        raise ValueError("Wideband bandwidth too small for the requested narrowband width.")
    if n_signals == 1:
        offsets = [float(rng.uniform(-available_bw / 2, available_bw / 2))]
    else:
        offsets = np.linspace(-available_bw / 2, available_bw / 2, n_signals)
        rng.shuffle(offsets)
    sig_list: List[NarrowbandSignal] = []
    wideband = np.zeros(num_samples, dtype=np.complex128)
    for k in range(n_signals):
        mod = usable[int(rng.integers(0, len(usable)))]
        pool = real_pool.get(mod, [])
        if pool:
            clip = pool[int(rng.integers(0, len(pool)))]
            nbs = tile_clip_to_length(clip, num_samples, rng)
            # Normalise then re-apply target-SNR noise for a fair sweep
            p = np.mean(np.abs(nbs) ** 2)
            if p > 0 and np.isfinite(p):
                nbs = nbs / np.sqrt(p)
            nbs = add_awgn(nbs, snr_db, rng)
        else:
            # Fallback (NOISE etc.): local synthesis
            nbs = generate_narrowband(num_samples, mod, snr_db, rng)
        sig_list.append(NarrowbandSignal(modulation=mod, samples=nbs,
                                         carrier_offset_hz=float(offsets[k]),
                                         start_sample=0, snr_db=snr_db))
        t = np.arange(num_samples) / fs
        wideband += nbs * np.exp(1j * 2 * np.pi * offsets[k] * t)
    wideband = add_awgn(wideband, snr_db, rng)
    return WidebandSample(samples=wideband, signals=sig_list, fs=fs, snr_db=snr_db)
