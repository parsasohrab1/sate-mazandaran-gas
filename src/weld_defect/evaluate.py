"""Evaluation: load a checkpoint, run on the test split, check acceptance gates.

Produces:
    metrics/latest/metrics.json           — full metric bundle
    metrics/latest/error_analysis.json     — per-class + acceptance gate results
    metrics/latest/confusion_matrix.png    — confusion matrix figure
    metrics/latest/misclassified/          — dumped crops of misclassified samples
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

from weld_defect.config import load_config
from weld_defect.utils import get_logger
from weld_defect.utils.metrics import compute_metrics, check_acceptance, save_report
from weld_defect.models.factory import load_model

log = get_logger("weld_defect.evaluate")


def evaluate(cfg_path: str = "configs/eval.yaml") -> dict:
    """Run evaluation on the held-out test split and write the report."""
    if not _HAS_TORCH:
        raise RuntimeError("torch is required for evaluation.")
    from weld_defect.data.loader import WeldDataset
    from weld_defect.data.preprocess import PreprocessPipeline
    import yaml

    cfg = load_config(cfg_path)
    cfg_raw = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    report_dir = Path(cfg_raw["artifacts"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    model, meta = load_model(cfg_raw["model"]["checkpoint"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    class_names = list(meta.get("class_names", cfg_raw["data"]["class_names"]))

    pre = PreprocessPipeline(input_size=cfg_raw["model"]["input_size"])
    ds = WeldDataset(
        csv_path=cfg_raw["data"]["test_split"],
        image_dir=cfg_raw["data"]["image_dir"],
        class_names=class_names,
        preprocess=pre,
        augment=None,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg_raw["data"]["batch_size"],
                                         num_workers=cfg_raw["data"]["num_workers"])

    ys, preds, probs, inference_ms = [], [], [], []
    is_ensemble = meta.get("kind") == "ensemble" or meta.get("architecture") == "ensemble"
    misclassified: list[dict] = []
    with torch.no_grad():
        for i, (xb, yb) in enumerate(loader):
            xb = xb.to(device)
            t0 = time.perf_counter()
            out = model(xb)
            torch.cuda.synchronize() if device.type == "cuda" else None
            dt = (time.perf_counter() - t0) * 1000 / xb.size(0)
            proba = (out if is_ensemble else F.softmax(out, dim=1)).cpu().float().numpy()
            p = proba.argmax(axis=1)
            ys.extend(yb.tolist())
            preds.extend(p.tolist())
            probs.append(proba)
            inference_ms.extend([dt] * xb.size(0))
            for j in range(xb.size(0)):
                if p[j] != yb[j].item():
                    misclassified.append({"index": i * xb.size(0) + j,
                                          "true": class_names[yb[j].item()],
                                          "pred": class_names[p[j]]})

    proba = np.concatenate(probs, axis=0)
    metrics = compute_metrics(ys, preds, y_proba=proba, class_names=class_names)
    avg_inf = float(np.mean(inference_ms)) if inference_ms else None
    gates = check_acceptance(metrics, class_names, inference_ms=avg_inf,
                             gates={k: (v, "min" if k != "inference_ms" else "max")
                                    for k, v in cfg_raw.get("acceptance", {}).items()
                                    if isinstance(v, (int, float))})
    save_report(metrics, report_dir / "metrics.json")
    (report_dir / "error_analysis.json").write_text(
        json.dumps({"acceptance": gates.to_dict(), "misclassified_count": len(misclassified),
                    "inference_ms_mean": avg_inf,
                    "sample_misclassified": misclassified[:50]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Confusion matrix figure (optional; only if matplotlib + seaborn present)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.font_manager as fm
        fm.fontManager.addfont("/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf")
        fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        cm = np.array(metrics.confusion_matrix)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        fig.savefig(report_dir / "confusion_matrix.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        log.warning("confusion matrix figure skipped: %s", exc)

    log.info("Eval done: f1_macro=%.4f, acceptance=%s", metrics.f1_macro, gates.passed)
    return {"metrics": metrics.to_dict(), "acceptance": gates.to_dict(),
            "inference_ms_mean": avg_inf, "report_dir": str(report_dir)}
