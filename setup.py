"""Packaging for wideband-signal-recognition."""
from setuptools import setup, find_packages

setup(
    name="wideband-signal-recognition",
    version="0.1.0",
    description="End-to-end deep learning framework for wideband signal recognition "
                "(adaptation of Vagollari et al., IEEE Access 2023)",
    author="Wideband Signal Recognition Project Contributors",
    license="MIT",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
        "scipy>=1.10",
        "matplotlib>=3.6",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "tqdm>=4.64",
        "pyyaml>=6.0",
        "torch>=2.0",
        "torchvision>=0.15",
        "Pillow>=9.0",
    ],
    extras_require={
        "notebook": ["ipython>=8.0", "ipykernel>=6.0", "jupyter>=1.0", "notebook>=6.5"],
        "test": ["pytest>=7.2", "pytest-cov>=4.0"],
        "yolo": ["ultralytics>=8.0.0"],
    },
)
