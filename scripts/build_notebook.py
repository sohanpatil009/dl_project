"""Helper to programmatically build the Jupyter notebooks.

This is a small utility that produces a valid `.ipynb` file from a list
of (cell_type, source) tuples. We use it to keep the notebook content
declarative and version-controllable.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import List, Tuple


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def nb_cells(*cells) -> dict:
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": [],
                "collapsed_sections": [],
                "machine_shape": "hm",
                "mount_dir": "/content",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for cell_type, source in cells:
        if isinstance(source, str):
            source = [source]
        nb["cells"].append({
            "cell_type": cell_type,
            "metadata": {"id": _uid()},
            "source": source,
            "execution_count": None,
            "outputs": [],
        })
    # Set code cell execution_count to None and outputs to []
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["execution_count"] = None
            c["outputs"] = []
    return nb


def md(text: str) -> Tuple[str, List[str]]:
    lines = text.strip().splitlines()
    return ("markdown", [l + "\n" for l in lines])


def code(text: str) -> Tuple[str, List[str]]:
    lines = text.strip("\n").splitlines()
    src = []
    for i, l in enumerate(lines):
        src.append(l + ("\n" if i < len(lines) - 1 else ""))
    return ("code", src)


def write_notebook(path: str, nb: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
