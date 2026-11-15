"""Sate Mazandaran Gas — AI Weld Defect Detection.

Petro Pala Toos / Dr. N. Ashrafi-Payaman.
CMMI V2.0 Maturity Level 5 (Optimizing).
"""

__version__ = "1.0.0"
__author__ = "Petro Pala Toos R&D / Dr. Nosratollah Ashrafi-Payaman"
__license__ = "Proprietary — Petro Pala Toos / Sate Mazandaran Gas"

# 6 classes per API 1104 / ASME Section V (see docs/DataDictionary.md).
CLASS_NAMES = [
    "no_defect",
    "porosity",
    "crack",
    "lack_of_fusion",
    "undercut",
    "slag_inclusion",
]

CLASS_NAMES_FA = {
    "no_defect": "بدون عیب",
    "porosity": "تخلخل",
    "crack": "ترک",
    "lack_of_fusion": "عدم ادغام",
    "undercut": "کم‌ادبی لبه",
    "slag_inclusion": "آخال سرباره",
}

# Focal-loss alpha from configs/train.yaml (counter class imbalance).
CLASS_WEIGHTS = [1.0, 1.4, 1.6, 1.5, 1.2, 1.3]
