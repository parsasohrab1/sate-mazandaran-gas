"""Training loop with MLflow tracking and early stopping.

Designed to be runnable both as a library and via the CLI. On a CPU-only
environment the loop automatically disables mixed precision and uses a small
batch size; the same code path runs on GPU when available.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

from weld_defect.config import TrainConfig, load_config
from weld_defect.utils import get_logger, seed_everything, ensure_dir
from weld_defect.utils.metrics import compute_metrics
from weld_defect.models.factory import build_from_config, save_model
from weld_defect.models.ensemble import EnsembleModel

log = get_logger("weld_defect.train")


class FocalLoss(nn.Module if _HAS_TORCH else object):
    """Multi-class focal loss for imbalanced defect classes."""

    def __init__(self, gamma: float = 2.0, alpha: list[float] | tuple[float, ...] | None = None) -> None:
        if _HAS_TORCH:
            super().__init__()
        self.gamma = gamma
        if alpha is not None and not isinstance(alpha, str):
            self.alpha = torch.tensor(list(alpha), dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, logits: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
        ce = F.cross_entropy(logits, target, reduction="none", weight=self.alpha.to(logits) if self.alpha is not None else None)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def _make_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    name = cfg.training.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg.training.lr, momentum=0.9, weight_decay=cfg.training.weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def _make_scheduler(optimizer, cfg):
    name = cfg.training.scheduler.lower()
    if name == "cosine_warm":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warm = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.training.warmup_epochs)
        cos = CosineAnnealingLR(optimizer, T_max=max(1, cfg.training.epochs - cfg.training.warmup_epochs))
        return SequentialLR(optimizer, schedulers=[warm, cos], milestones=[cfg.training.warmup_epochs])
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return None


def _epoch(model, loader, criterion, optimizer, device, train: bool, scaler=None):
    """Run one epoch. Returns a metrics dict (including ``loss``).

    ``scaler`` is accepted for API symmetry but AMP is handled via autocast
    context below; the explicit scale/unscale/step dance is intentionally
    avoided to keep the loop readable and testable on CPU.
    """
    is_ensemble = isinstance(model, EnsembleModel)
    model.train() if train else model.eval()
    ys, preds, probs, total_loss = [], [], [], 0.0
    use_amp = scaler is not None and train and torch.cuda.is_available()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(xb)
                loss = criterion(torch.log(out.clamp(min=1e-8)), yb) if is_ensemble else criterion(out, yb)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += float(loss.item()) * xb.size(0)
            ys.extend(yb.cpu().tolist())
            proba = (out if is_ensemble else F.softmax(out, dim=1)).cpu().detach().float().numpy()
            preds.extend(proba.argmax(axis=1).tolist())
            probs.append(proba)

    y = np.asarray(ys)
    proba = np.concatenate(probs, axis=0) if probs else np.zeros((0, 6))
    metrics = compute_metrics(y, np.asarray(preds), y_proba=proba if proba.size else None)
    metrics_dict = metrics.to_dict()
    metrics_dict["loss"] = total_loss / max(1, len(y))
    return metrics_dict


def train(cfg_path: str = "configs/train.yaml") -> dict:
    """Run a full training pass and return the best metrics + artifact paths."""
    if not _HAS_TORCH:
        raise RuntimeError("torch is required for training. Install requirements.txt first.")
    cfg = load_config(cfg_path, TrainConfig)
    seed_everything(cfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on device=%s, experiment=%s", device, cfg.experiment.name)

    from weld_defect.data.loader import build_dataloaders

    train_loader, val_loader, _ = build_dataloaders(cfg)
    model = build_from_config(cfg.model).to(device)

    criterion = FocalLoss(gamma=cfg.loss.gamma, alpha=list(cfg.loss.alpha))
    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.mixed_precision and torch.cuda.is_available())

    # MLflow tracking (optional — degrades to local logging if unavailable)
    mlflow = None
    try:
        import mlflow
        mlflow.set_tracking_uri(cfg.experiment.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.experiment.mlflow_experiment)
        mlflow.start_run(run_name=cfg.experiment.name)
        mlflow.log_params(asdict(cfg.training))
        mlflow.log_params({"architecture": cfg.model.architecture, "members": ",".join(cfg.model.members)})
    except Exception as exc:  # pragma: no cover
        log.warning("MLflow disabled: %s", exc)

    best_metric, best_state, patience_left = -1.0, None, cfg.early_stopping.patience
    ensure_dir(cfg.artifacts.checkpoint_dir)
    history: list[dict] = []

    for epoch in range(1, cfg.training.epochs + 1):
        t0 = time.time()
        tr = _epoch(model, train_loader, criterion, optimizer, device, train=True, scaler=scaler)
        va = _epoch(model, val_loader, criterion, optimizer, device, train=False)
        if scheduler is not None:
            scheduler.step()
        log.info("epoch %02d | train_f1=%.3f val_f1=%.3f val_loss=%.4f | %.1fs",
                 epoch, tr["f1_macro"], va["f1_macro"], va["loss"], time.time() - t0)
        history.append({"epoch": epoch, "train": tr, "val": va})
        if mlflow is not None:
            mlflow.log_metric("train_f1", tr["f1_macro"], step=epoch)
            mlflow.log_metric("val_f1", va["f1_macro"], step=epoch)
            mlflow.log_metric("val_loss", va["loss"], step=epoch)
        # early stopping (maximize macro-F1)
        if va["f1_macro"] > best_metric:
            best_metric = va["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.early_stopping.patience
        else:
            patience_left -= 1
            if cfg.early_stopping.enabled and patience_left <= 0:
                log.info("Early stopping at epoch %d (best val_f1=%.3f)", epoch, best_metric)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = Path(cfg.artifacts.export_dir) / "ensemble_best.pt"
    save_model(model, ckpt, meta={
        "architecture": cfg.model.architecture,
        "members": cfg.model.members,
        "num_classes": cfg.data.num_classes,
        "input_size": cfg.data.input_size,
        "class_names": list(cfg.data.class_names),
        "val_f1_macro": best_metric,
    })
    if mlflow is not None:
        try:
            mlflow.log_artifact(str(ckpt))
        except Exception:
            pass
        mlflow.end_run()

    log.info("Done. best val_f1_macro=%.4f, checkpoint=%s", best_metric, ckpt)
    return {"best_val_f1_macro": best_metric, "checkpoint": str(ckpt), "history": history}
