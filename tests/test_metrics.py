"""Metrics + acceptance-gate tests — scikit-learn only (no torch)."""
from __future__ import annotations

import numpy as np

from weld_defect.utils.metrics import (
    compute_metrics, check_acceptance, ClassificationMetrics, AcceptanceResult,
)


def _perfect_case():
    y = [0, 1, 2, 3, 4, 5]
    return y, y


def test_compute_metrics_perfect():
    y, p = _perfect_case()
    m = compute_metrics(y, p, class_names=["c0","c1","c2","c3","c4","c5"])
    assert isinstance(m, ClassificationMetrics)
    assert m.accuracy == 1.0
    assert m.f1_macro == 1.0
    assert m.balanced_accuracy == 1.0
    assert len(m.confusion_matrix) == 6


def test_compute_metrics_with_proba():
    y = [0, 1, 2]
    p = [0, 1, 2]
    proba = np.eye(3)[y]  # one-hot
    m = compute_metrics(y, p, y_proba=proba, class_names=["a","b","c"])
    assert m.roc_auc_ovr == 1.0


def test_acceptance_passes_on_perfect():
    y, p = _perfect_case()
    names = ["no_defect","porosity","crack","lack_of_fusion","undercut","slag_inclusion"]
    m = compute_metrics(y, p, class_names=names)
    res = check_acceptance(m, names, inference_ms=10.0)
    assert isinstance(res, AcceptanceResult)
    assert res.passed is True
    assert all(c["passed"] for c in res.checks)


def test_acceptance_fails_low_crack_recall():
    # crack is index 2; predict everything as no_defect
    y = [0, 1, 2, 3, 4, 5]
    p = [0, 0, 0, 0, 0, 0]
    names = ["no_defect","porosity","crack","lack_of_fusion","undercut","slag_inclusion"]
    m = compute_metrics(y, p, class_names=names)
    res = check_acceptance(m, names, inference_ms=10.0)
    assert res.passed is False
    crack_gate = next(c for c in res.checks if c["name"] == "recall_crack")
    assert crack_gate["passed"] is False


def test_acceptance_fails_high_latency():
    y, p = _perfect_case()
    names = ["no_defect","porosity","crack","lack_of_fusion","undercut","slag_inclusion"]
    m = compute_metrics(y, p, class_names=names)
    res = check_acceptance(m, names, inference_ms=500.0)
    assert res.passed is False
    lat = next(c for c in res.checks if c["name"] == "inference_ms")
    assert lat["passed"] is False


def test_metrics_to_dict_roundtrip():
    y, p = _perfect_case()
    m = compute_metrics(y, p, class_names=["a","b","c","d","e","f"])
    d = m.to_dict()
    assert set(d.keys()) >= {"accuracy", "f1_macro", "confusion_matrix", "per_class"}
