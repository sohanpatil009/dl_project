# Data directory

This directory holds **generated or downloaded** datasets. Nothing large is
committed to git (see `.gitignore`).

## Layout expected by the code

```
data/
├── yolo_dataset/            # spectrogram images + YOLO labels
│   ├── images/{train,val,test}/*.{jpg,png,bmp}
│   ├── labels/{train,val,test}/*.txt
│   └── data.yaml
├── narrowband/              # IQ clips for the classifier
│   ├── BPSK/*.npy
│   ├── QPSK/*.npy
│   ├── 8PSK/*.npy
│   ├── 16QAM/*.npy
│   ├── 64QAM/*.npy
│   ├── NOISE/*.npy
│   ├── NO_SIGNAL/*.npy
│   └── manifest.json
└── public/                  # downloaded internet data (RadioML, git-ignored)
    └── RML2016.10a_dict.pkl
```

## How to create it (synthetic, built-in)

From the project root:

```bash
pip install -r requirements.txt
python scripts/prepare_dataset.py --wideband-train 600 --narrowband-per-class 400
# PNG instead of JPG:
python scripts/prepare_dataset.py --image-format png --wideband-train 600
```

Use smaller numbers for a quick smoke test:

```bash
python scripts/prepare_dataset.py --wideband-train 30 --narrowband-per-class 40
```

## How to use REAL internet data (both stages) — recommended

We wire in **RadioML 2016.10A** (DeepSig, CC BY-NC-SA 4.0): 11 modulations ×
20 SNRs × 1000 clips of 2×128 I/Q with realistic impairments. The 5 overlapping
classes (BPSK/QPSK/8PSK/QAM16→16QAM/QAM64→64QAM) back the classifier directly;
wideband captures mix tiled real clips (documented adaptation); NOISE/NO_SIGNAL
are synthesised locally.

```bash
# Colab / local (downloads the ~213 MB Zenodo mirror into data/public/)
python scripts/fetch_public_data.py --out data --narrowband-per-class 400 \
    --wideband-train 600 --image-format jpg

# Kaggle (attach an RML2016.10a mirror via + Add Data, skip download)
python scripts/fetch_public_data.py --out /kaggle/working/wsr \
    --radioml-pkl /kaggle/input/<dataset>/RML2016.10a_dict.pkl \
    --image-format jpg --wideband-train 300
```

Or through the unified entry point:

```bash
python scripts/prepare_dataset.py --data-source radioml --image-format png
```

## Using your own data

If the original Vagollari et al. dataset (or your own captures) becomes
available, place it in `data/yolo_dataset/` using the layout above and add a
matching `data.yaml`. `src/data_loader.discover_external_dataset()` will pick
it up automatically.

## How the public data differs from the paper (read before comparing numbers)

| Aspect | Paper (Vagollari et al.) | This repo + RadioML |
|---|---|---|
| Modulations | BPSK/QPSK/8PSK/16QAM/64QAM + NOISE + NO_SIGNAL | 5 overlap exactly; AM/CPFSK/GFSK/PAM4/WBFM available opt-in (`--include-extra`); NOISE/NO_SIGNAL synthesised |
| Clip length | paper-specific | 2×128 (native RadioML = native `clf_seq_len`) |
| Train SNR | 20 dB | nearest RadioML bin: 18 dB (`radioml_snr_db`) |
| Wideband | original Sims | tiled real clips + same mixing/geometry |
| Channel | paper's model | real GNU-Radio impairments (fading, offsets) |

## Sources / licence

* Zenodo mirror: DOI `10.5281/zenodo.18397070` (Colab/local download).
* Kaggle mirrors: `zaslee/rml2016-10a`, `nolasthitnotomorrow/radioml2016-deepsigcom` (+ Add Data).
* Licence: CC BY-NC-SA 4.0 (non-commercial research). Cite O'Shea & West (2016).
