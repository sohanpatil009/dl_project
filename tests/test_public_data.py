"""Unit tests for the public internet-data module (no download needed)."""
import numpy as np
import pytest

from src.public_data import (
    OVERLAP_MAP, EXTRA_MODS, RADIOML_MODS,
    radioml_to_clips, tile_clip_to_length,
    generate_wideband_from_real,
)


def _mock_radioml(n_per_bin: int = 8) -> dict:
    rng = np.random.default_rng(0)
    data = {}
    for mod in RADIOML_MODS:
        for snr in (-4, 0, 18):
            arr = rng.normal(0, 1, (n_per_bin, 2, 128)).astype(np.float32)
            data[(mod, snr)] = arr
    return data


def test_overlap_map_covers_paper_classes():
    assert set(OVERLAP_MAP.values()) == {"BPSK", "QPSK", "8PSK", "16QAM", "64QAM"}
    assert len(set(EXTRA_MODS) & set(OVERLAP_MAP)) == 0


def test_radioml_to_clips_shapes():
    rng = np.random.default_rng(1)
    data = _mock_radioml()
    clips = radioml_to_clips(data, snr_db=18.0, per_class=4, rng=rng)
    assert set(clips) == set(OVERLAP_MAP.values())
    for c in clips.values():
        assert len(c) == 4
        for clip in c:
            assert clip.shape == (128,)
            assert np.iscomplexobj(clip)


def test_radioml_to_clips_nearest_snr():
    rng = np.random.default_rng(2)
    data = _mock_radioml()
    # 20 dB not in mock bins (-4,0,18) -> nearest (18) must be used, not crash
    clips = radioml_to_clips(data, snr_db=20.0, per_class=2, rng=rng)
    assert len(clips) == 5


def test_radioml_to_clips_include_extra():
    rng = np.random.default_rng(3)
    data = _mock_radioml()
    clips = radioml_to_clips(data, snr_db=0.0, per_class=2, rng=rng,
                             include_extra=True)
    assert "WBFM" in clips  # extra class kept under its own name


def test_tile_clip_to_length():
    rng = np.random.default_rng(4)
    clip = (np.random.randn(128) + 1j * np.random.randn(128))
    tiled = tile_clip_to_length(clip, 1000, rng)
    assert tiled.shape == (1000,)
    # Tiled signal is periodic with period 128 (random start offset allowed)
    assert np.allclose(tiled[:872], tiled[128:1000])
    # All tiled values come from the original clip
    assert set(np.round(tiled, 8)).issubset(set(np.round(clip, 8)))


def test_generate_wideband_from_real():
    rng = np.random.default_rng(5)
    data = _mock_radioml()
    pool = radioml_to_clips(data, snr_db=0.0, per_class=6,
                            rng=np.random.default_rng(6))
    # add a NOISE fallback entry missing from pool on purpose
    s = generate_wideband_from_real(
        num_samples=8000, fs=200_000.0, snr_db=10.0, real_pool=pool,
        mod_classes=["BPSK", "QPSK", "16QAM", "NOISE"],
        min_signals=1, max_signals=2,
        bandwidth=12_000.0, padding=2_000.0, rng=rng)
    assert 1 <= len(s.signals) <= 2
    assert s.samples.shape == (8000,)


def test_image_formats_supported():
    from src.dataset_utils import IMAGE_FORMATS
    assert set(IMAGE_FORMATS) >= {"jpg", "png", "bmp"}


def test_save_wideband_dataset_png(tmp_path):
    from src.preprocessing import generate_wideband
    from src.dataset_utils import save_wideband_dataset
    rng = np.random.default_rng(7)
    s = generate_wideband(num_samples=8000, fs=200_000.0, snr_db=20.0,
                          mod_classes=["BPSK", "QPSK"], min_signals=1,
                          max_signals=1, bandwidth=12_000.0,
                          padding=2_000.0, rng=rng)
    stored = save_wideband_dataset([s], out_dir=str(tmp_path), spec_image_size=128,
                                   n_fft=256, hop_length=64, image_format="png")
    assert stored[0].spec_path.endswith(".png")
    import os
    assert os.path.isfile(stored[0].spec_path)
