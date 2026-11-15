"""Metric computation and acceptance-gate checks.

Wraps scikit-learn metrics in a stable, importable API and adds the
project-specific safety gates defined in ``configs/eval.yaml`` (e.g. recall on
the ``crack`` class must be ≥ 0.95).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - sklearn is a hard dep but be safe in CI
    _HAS_SKLEARN = False


@dataclass
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    cohen_kappa: float
    matthews_corrcoef: float
    roc_auc_ovr: float | None
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "f1_macro": self.f1_macro,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "cohen_kappa": self.cohen_kappa,
            "matthews_corrcoef": self.matthews_corrcoef,
            "roc_auc_ovr": self.roc_auc_ovr,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
        }


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: np.ndarray | None = None,
    class_names: Sequence[str] | None = None,
) -> ClassificationMetrics:
    """Compute the full metric bundle for one evaluation pass."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if not _HAS_SKLEARN:
        acc = float((y_true == y_pred).mean())
        return ClassificationMetrics(
            accuracy=acc, balanced_accuracy=acc, f1_macro=acc,
            precision_macro=acc, recall_macro=acc, cohen_kappa=0.0,
            matthews_corrcoef=0.0, roc_auc_ovr=None,
        )

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    names = list(class_names) if class_names else [str(i) for i in range(len(per_class_p))]
    per_class = {
        names[i]: {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(per_class_p))
    }
    auc: float | None = None
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except ValueError:
            auc = None
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        precision_macro=float(p),
        recall_macro=float(r),
        cohen_kappa=float(cohen_kappa_score(y_true, y_pred)),
        matthews_corrcoef=float(matthews_corrcoef(y_true, y_pred)),
        roc_auc_ovr=auc,
        per_class=per_class,
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
    )


@dataclass
class AcceptanceResult:
    passed: bool
    checks: list[dict]  # each: {name, threshold, actual, passed}

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": self.checks}


# Default gates mirror configs/eval.yaml → acceptance section.
DEFAULT_GATES = {
    "f1_macro": (0.85, "min"),
    "recall_crack": (0.95, "min"),
    "precision_no_defect": (0.90, "min"),
    "inference_ms": (120.0, "max"),
}


def check_acceptance(
    metrics: ClassificationMetrics,
    class_names: Sequence[str],
    inference_ms: float | None = None,
    gates: dict | None = None,
) -> AcceptanceResult:
    """Evaluate the BRS acceptance gates; returns pass/fail per gate."""
    gates = gates or DEFAULT_GATES
    idx = {n: i for i, n in enumerate(class_names)}
    crack_recall = metrics.per_class.get("crack", {}).get("recall", 0.0)
    no_def_prec = metrics.per_class.get("no_defect", {}).get("precision", 0.0)

    candidates = {
        "f1_macro": metrics.f1_macro,
        "recall_crack": crack_recall,
        "precision_no_defect": no_def_prec,
        "inference_ms": inference_ms if inference_ms is not None else float("inf"),
    }
    checks: list[dict] = []
    ok = True
    for name, (threshold, mode) in gates.items():
        actual = candidates.get(name, None)
        if actual is None:
            checks.append({"name": name, "threshold": threshold, "actual": None, "passed": False})
            ok = False
            continue
        passed = (actual >= threshold) if mode == "min" else (actual <= threshold)
        ok = ok and passed
        checks.append({
            "name": name,
            "threshold": threshold,
            "actual": float(actual),
            "mode": mode,
            "passed": bool(passed),
        })
    return AcceptanceResult(passed=bool(ok), checks=checks)


def save_report(metrics: ClassificationMetrics, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return p
