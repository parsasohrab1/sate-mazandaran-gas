"""End-to-end integration test — synthetic data → train (1 epoch) → evaluate.

Marked ``integration`` + ``slow`` so it is excluded from the fast CI unit pass
and run in the dedicated ML validation job (``.github/workflows/ml.yml``).
Requires torch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.integration, pytest.mark.slow]

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_synthetic_data import generate


def _make_ci_config(repo_root: Path, out: Path) -> tuple[Path, Path]:
    """Produce tiny train + eval configs pointing at synthetic data."""
    import yaml
    train_cfg = yaml.safe_load((repo_root / "configs" / "train.yaml").read_text(encoding="utf-8"))
    train_cfg["training"]["epochs"] = 1
    train_cfg["training"]["batch_size"] = 8
    train_cfg["training"]["num_workers"] = 0
    train_cfg["data"]["input_size"] = 64
    train_cfg["data"]["image_dir"] = str(out / "images")
    train_cfg["data"]["train_split"] = str(out / "train.csv")
    train_cfg["data"]["val_split"] = str(out / "val.csv")
    train_cfg["data"]["test_split"] = str(out / "test.csv")
    train_cfg["model"]["members"] = ["custom_cnn"]
    train_cfg["artifacts"]["export_dir"] = str(out / "models")
    train_cfg["artifacts"]["checkpoint_dir"] = str(out / "models" / "checkpoints")
    train_cfg["experiment"]["mlflow_tracking_uri"] = f"file:{out}/mlruns"
    tr_path = out / "train_ci.yaml"
    tr_path.write_text(yaml.dump(train_cfg, allow_unicode=True), encoding="utf-8")

    eval_cfg = yaml.safe_load((repo_root / "configs" / "eval.yaml").read_text(encoding="utf-8"))
    eval_cfg["model"]["checkpoint"] = str(out / "models" / "ensemble_best.pt")
    eval_cfg["model"]["input_size"] = 64
    eval_cfg["data"]["test_split"] = str(out / "test.csv")
    eval_cfg["data"]["image_dir"] = str(out / "images")
    eval_cfg["data"]["batch_size"] = 8
    eval_cfg["data"]["num_workers"] = 0
    eval_cfg["artifacts"]["report_dir"] = str(out / "metrics")
    # Loosen gates for the tiny smoke run (still recorded).
    eval_cfg["acceptance"]["min_f1_macro"] = 0.05
    eval_cfg["acceptance"]["min_recall_crack"] = 0.05
    eval_cfg["acceptance"]["min_precision_no_defect"] = 0.05
    eval_cfg["acceptance"]["max_inference_ms"] = 10000
    ev_path = out / "eval_ci.yaml"
    ev_path.write_text(yaml.dump(eval_cfg, allow_unicode=True), encoding="utf-8")
    return tr_path, ev_path


def test_synthetic_train_eval_e2e(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    out = tmp_path / "run"
    (out / "models").mkdir(parents=True, exist_ok=True)
    # 1) generate synthetic data (small)
    generate(n_samples=80, size=64, seed=42, out=out, n_groups=8)
    # 2) build CI configs
    tr_cfg, ev_cfg = _make_ci_config(repo_root, out)
    # 3) train
    from weld_defect.train import train as _train
    res = _train(str(tr_cfg))
    assert Path(res["checkpoint"]).exists()
    # 4) evaluate
    from weld_defect.evaluate import evaluate as _eval
    ev = _eval(str(ev_cfg))
    assert ev["metrics"]["f1_macro"] >= 0.0
    assert "acceptance" in ev
    assert "checks" in ev["acceptance"]
