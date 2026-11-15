"""Sate Mazandaran Gas — AI Weld Defect Detection package.

A decision-support system for detecting and classifying weld defects in
pipeline radiographs, commissioned by Sate Mazandaran Gas (شرکت گاز استان
مازندران) and executed by Petro Pala Toos (شرکت پترو پالا توس).

The package is structured around five concerns:

    weld_defect.data     — loading, preprocessing, augmentation, splits
    weld_defect.models   — CNN, ResNet, EfficientNet, and the production ensemble
    weld_defect.train    — training loop with MLflow tracking
    weld_defect.evaluate — metrics, confusion matrix, error analysis
    weld_defect.api      — FastAPI inference service

Safety note (see SRS §5.4 and the LICENSE):
    Model outputs are decision-support ("پشتیبانی از کارشناس") and must be
    reviewed by a certified NDT specialist per API 1104 / ASME Section V before
    any safety-critical decision is taken.
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__", "__author__", "__project__"]

__project__ = "sate-mazandaran-gas"
__author__ = "Petro Pala Toos R&D / Dr. N. Ashrafi-Payaman"

try:
    __version__ = version("sate-mazandaran-gas")
except PackageNotFoundError:  # editable/dev install
    __version__ = "1.0.0"

# Canonical class catalogue — single source of truth, imported by configs.
CLASS_NAMES = (
    "no_defect",
    "porosity",
    "crack",
    "lack_of_fusion",
    "undercut",
    "slag_inclusion",
)
NUM_CLASSES = len(CLASS_NAMES)
