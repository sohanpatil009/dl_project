"""Thin CLI wrapper: train the modulation classifier.

Usage:
    python scripts/train_classifier.py --data data/narrowband --arch resnet18
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train_classifier import main

if __name__ == "__main__":
    raise SystemExit(main())
