"""Metric computation and acceptance-gate checking.

Implements SRS FR-3 (Evaluation and Validation) and BRS §6 (Acceptance Criteria).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def compute_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    y_prob: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute the full metric suite per SRS FR-3.1.

    Returns a dict with: accuracy, balanced_accuracy, precision_macro,
    recall_macro, f1_macro, cohen_kappa, matthews_corrcoef, roc_auc_ovr
    (if ``y_prob`` provided), confusion_matrix, per_class.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
    }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))) if class_names else confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    per_class = []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(cm.sum() - tp - fp - fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class.append({
            "class_name": name,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(cm[i, :].sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    metrics["per_class"] = per_class

    if y_prob is not None:
        try:
            from sklearn.metrics import roc_auc_score
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        except Exception:
            metrics["roc_auc_ovr"] = None

    return metrics


def check_acceptance(metrics: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    """Compare metrics against BRS §6 acceptance gates.

    ``gates`` keys: min_f1_macro, min_recall_crack, min_precision_no_defect,
    max_inference_ms, require_expert_review.
    """
    per_class = {pc["class_name"]: pc for pc in metrics.get("per_class", [])}
    inference_ms = metrics.get("inference_ms_avg") or metrics.get("inference_ms") or 0.0

    checks: dict[str, dict[str, Any]] = {}

    checks["f1_macro"] = {
        "value": metrics.get("f1_macro", 0.0),
        "threshold": gates.get("min_f1_macro", 0.85),
        "pass": metrics.get("f1_macro", 0.0) >= gates.get("min_f1_macro", 0.85),
    }

    crack_recall = per_class.get("crack", {}).get("recall", 0.0)
    checks["recall_crack"] = {
        "value": crack_recall,
        "threshold": gates.get("min_recall_crack", 0.95),
        "pass": crack_recall >= gates.get("min_recall_crack", 0.95),
    }

    no_defect_precision = per_class.get("no_defect", {}).get("precision", 0.0)
    checks["precision_no_defect"] = {
        "value": no_defect_precision,
        "threshold": gates.get("min_precision_no_defect", 0.90),
        "pass": no_defect_precision >= gates.get("min_precision_no_defect", 0.90),
    }

    checks["inference_ms"] = {
        "value": inference_ms,
        "threshold": gates.get("max_inference_ms", 120),
        "pass": inference_ms <= gates.get("max_inference_ms", 120),
    }

    checks["require_expert_review"] = {
        "value": gates.get("require_expert_review", True),
        "threshold": True,
        "pass": gates.get("require_expert_review", True) is True,
    }

    overall_pass = all(c["pass"] for c in checks.values())
    return {"checks": checks, "pass": overall_pass}


def save_metrics_report(metrics: dict[str, Any], out_dir: str | Path) -> Path:
    """Write metrics.json, confusion_matrix.png, per_class_report.txt to ``out_dir``."""
    from .io import ensure_dir, save_json

    out = ensure_dir(out_dir)

    # metrics.json
    save_json(metrics, out / "metrics.json")

    # confusion_matrix.png
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        cm = np.array(metrics.get("confusion_matrix", []))
        if cm.size:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix")
            if cm.shape[0] <= 10:
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                                color="white" if cm[i, j] > cm.max() / 2 else "black")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(out / "confusion_matrix.png", dpi=100)
            plt.close(fig)
    except Exception:
        pass

    # per_class_report.txt
    per_class = metrics.get("per_class", [])
    lines = ["Per-class report", "=" * 60]
    for pc in per_class:
        lines.append(
            f"{pc['class_name']:>20s} | P={pc['precision']:.3f} R={pc['recall']:.3f} "
            f"F1={pc['f1']:.3f} support={pc['support']}"
        )
    (out / "per_class_report.txt").write_text("\n".join(lines), encoding="utf-8")

    return out / "metrics.json"
