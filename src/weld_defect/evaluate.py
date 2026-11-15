"""Evaluation — metrics, acceptance gates, error analysis.

Implements SRS FR-3 (evaluation and validation).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import EvalConfig, load_eval_config
from .data.augment import get_val_aug
from .data.loader import WeldDataset
from .models.factory import load_model
from .utils.io import ensure_dir, save_json
from .utils.metrics import check_acceptance, compute_metrics, save_metrics_report


def evaluate(config_path: str = "configs/eval.yaml") -> dict[str, Any]:
    """Main evaluation entry point.

    Args:
        config_path: Path to the eval config YAML.

    Returns:
        Dict with metrics, acceptance, and report paths.
    """
    config = load_eval_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model.
    try:
        model = load_model(
            config.model.checkpoint,
            architecture=config.model.architecture,
            num_classes=config.model.num_classes,
        ).to(device)
        model_loaded = True
    except Exception as e:
        print(f"Warning: could not load checkpoint ({e}); using untrained model.")
        from .models.factory import build_model
        model = build_model(config.model.architecture, num_classes=config.model.num_classes, pretrained=False).to(device)
        model_loaded = False

    # Build test loader.
    val_aug = get_val_aug(config.model.input_size)
    label_to_idx = {name: i for i, name in enumerate(config.data.class_names)}
    test_ds = WeldDataset(config.data.test_split, config.data.image_dir, val_aug, config.model.input_size, label_to_idx)
    test_loader = DataLoader(test_ds, batch_size=config.data.batch_size, shuffle=False, num_workers=config.data.num_workers)

    # Run inference + measure latency.
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[np.ndarray] = []
    misclassified: list[dict[str, Any]] = []
    inference_times: list[float] = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            t0 = time.perf_counter()
            logits = model(imgs)
            t1 = time.perf_counter()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())

            batch_time_ms = (t1 - t0) * 1000 / max(1, imgs.shape[0])
            inference_times.append(batch_time_ms)

            # Track misclassified for CAR analysis.
            for i, (true, pred) in enumerate(zip(labels.numpy(), preds)):
                if true != pred:
                    row = test_ds.manifest.iloc[len(y_true) - len(labels) + i]
                    misclassified.append({
                        "image": row["image"],
                        "true_class": test_ds.idx_to_label[int(true)],
                        "predicted_class": test_ds.idx_to_label[int(pred)],
                        "confidence": float(probs[i].max()),
                    })

    # Compute metrics.
    metrics = compute_metrics(
        np.array(y_true), np.array(y_pred),
        y_prob=np.array(y_prob) if y_prob else None,
        class_names=config.data.class_names,
    )
    metrics["inference_ms_avg"] = float(np.mean(inference_times)) if inference_times else 0.0
    metrics["model"] = config.model.architecture
    metrics["checkpoint"] = config.model.checkpoint
    metrics["test_samples"] = len(y_true)
    metrics["model_loaded"] = model_loaded

    # Acceptance gates.
    acceptance_gates = {
        "min_f1_macro": config.acceptance.min_f1_macro,
        "min_recall_crack": config.acceptance.min_recall_crack,
        "min_precision_no_defect": config.acceptance.min_precision_no_defect,
        "max_inference_ms": config.acceptance.max_inference_ms,
        "require_expert_review": config.acceptance.require_expert_review,
    }
    acceptance = check_acceptance(metrics, acceptance_gates)

    # Save report.
    report_dir = ensure_dir(config.artifacts.report_dir)
    save_metrics_report(metrics, report_dir)

    error_analysis = {
        "misclassified_count": len(misclassified),
        "misclassified": misclassified[:100],  # cap for report size
        "acceptance": acceptance,
    }
    save_json(error_analysis, config.artifacts.error_analysis_json)

    result = {
        "metrics": metrics,
        "acceptance": acceptance,
        "report_dir": str(report_dir),
        "error_analysis_json": str(config.artifacts.error_analysis_json),
    }
    print(f"Evaluation complete. Acceptance: {'PASS' if acceptance['pass'] else 'FAIL'}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate weld defect detection model")
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()
    evaluate(args.config)
