"""Training loop for weld defect detection.

Implements SRS FR-2 (training) with:
  - focal loss (FR-2.2)
  - AdamW + cosine warmup (FR-2.2)
  - early stopping on val_f1_macro (FR-2.4)
  - MLflow tracking (FR-2.3, lazy)
  - reproducibility via seed_everything (FR-2.5)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import TrainConfig, load_config
from .data.augment import get_train_aug, get_val_aug
from .data.loader import WeldDataset
from .models.factory import build_model, save_model
from .utils.io import ensure_dir, save_json
from .utils.metrics import compute_metrics
from .utils.seed import seed_everything


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default 2.0).
        alpha: Per-class weights (default CLASS_WEIGHTS from train.yaml).
        reduction: 'mean' or 'sum' or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: list[float] | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha.to(logits.device) if self.alpha is not None else None, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def _try_start_mlflow(config: TrainConfig) -> Any:
    """Start an MLflow run if mlflow is installed; return a context manager or None."""
    try:
        import mlflow
        mlflow.set_tracking_uri(config.experiment.mlflow_tracking_uri)
        mlflow.set_experiment(config.experiment.mlflow_experiment)
        run = mlflow.start_run(run_name=config.experiment.name)
        mlflow.log_params({
            "architecture": config.model.architecture,
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "lr": config.training.lr,
            "optimizer": config.training.optimizer,
            "loss": config.loss.name,
            "gamma": config.loss.gamma,
            "seed": config.experiment.seed,
        })
        return run
    except Exception:
        return None


def _try_log_mlflow(metrics: dict[str, float], step: int = 0) -> None:
    try:
        import mlflow
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}, step=step)
    except Exception:
        pass


def _try_end_mlflow(run: Any) -> None:
    if run is not None:
        try:
            run.__exit__(None, None, None)
        except Exception:
            pass


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, class_names: list[str]) -> dict[str, Any]:
    """Run evaluation on a loader; return metrics dict."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())
    return compute_metrics(np.array(y_true), np.array(y_pred), class_names=class_names)


def train(config_path: str = "configs/train.yaml") -> dict[str, Any]:
    """Main training entry point.

    Args:
        config_path: Path to the training config YAML.

    Returns:
        Dict with final metrics and checkpoint path.
    """
    config = load_config(config_path)
    seed_everything(config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- data ---
    label_to_idx = {name: i for i, name in enumerate(config.data.class_names)}
    train_aug = get_train_aug(config.augmentation.ops, config.data.input_size) if config.augmentation.enabled else get_val_aug(config.data.input_size)
    val_aug = get_val_aug(config.data.input_size)

    train_ds = WeldDataset(config.data.train_split, config.data.image_dir, train_aug, config.data.input_size, label_to_idx)
    val_ds = WeldDataset(config.data.val_split, config.data.image_dir, val_aug, config.data.input_size, label_to_idx)

    train_loader = DataLoader(
        train_ds, batch_size=config.training.batch_size, shuffle=True,
        num_workers=config.training.num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.training.batch_size, shuffle=False,
        num_workers=config.training.num_workers,
    )

    # --- model ---
    model = build_model(
        config.model.architecture,
        num_classes=config.data.num_classes,
        pretrained=config.model.pretrained,
        config={
            "members": config.model.members,
            "ensemble_strategy": config.model.ensemble_strategy,
        },
        dropout=config.model.dropout,
    ).to(device)

    # --- loss / optim / scheduler ---
    criterion = FocalLoss(gamma=config.loss.gamma, alpha=config.loss.alpha)

    # SAM (Sharpness-Aware Minimization) optimizer — enabled via config flag.
    use_sam = config.raw.get("training", {}).get("use_sam", False) if hasattr(config, "raw") else False
    if use_sam:
        from .sam import SAM
        optimizer = SAM(
            model.parameters(),
            base_optimizer_cls=AdamW,
            rho=0.05,
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
        print("Using SAM optimizer (rho=0.05)")
    else:
        optimizer = AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)

    # Cosine scheduler — works with both AdamW and SAM (wraps base_optimizer).
    scheduler = CosineAnnealingLR(
        optimizer.base_optimizer if use_sam else optimizer,
        T_max=max(1, config.training.epochs - config.training.warmup_epochs),
    )

    # --- mlflow ---
    mlflow_run = _try_start_mlflow(config)

    # --- training loop ---
    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    history: list[dict[str, Any]] = []

    ensure_dir(config.artifacts.checkpoint_dir)
    ensure_dir(config.artifacts.export_dir)

    for epoch in range(config.training.epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_sam:
                # SAM: two-step optimization.
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                # Second forward pass at perturbed weights.
                logits2 = model(imgs)
                loss2 = criterion(logits2, labels)
                loss2.backward()
                if config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
                optimizer.second_step(zero_grad=True)
                running_loss += float(loss.item())
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                if config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
                optimizer.step()
                running_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        avg_loss = running_loss / max(1, n_batches)

        val_metrics = _evaluate(model, val_loader, device, config.data.class_names)
        val_f1 = val_metrics["f1_macro"]

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_f1_macro": val_f1,
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(epoch_info)
        print(f"Epoch {epoch+1}/{config.training.epochs} | loss={avg_loss:.4f} | val_f1={val_f1:.4f}")

        _try_log_mlflow(epoch_info, step=epoch)

        # Early stopping.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if config.training.early_stopping.enabled and patience_counter >= config.training.early_stopping.patience:
                print(f"Early stopping at epoch {epoch+1} (patience={config.training.early_stopping.patience})")
                break

    # Restore best.
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint.
    ckpt_path = Path(config.artifacts.export_dir) / "ensemble_best.pt"
    if config.model.architecture == "ensemble":
        ckpt_path = Path(config.artifacts.export_dir) / "ensemble_best.pt"
    else:
        ckpt_path = Path(config.artifacts.export_dir) / f"{config.model.architecture}_best.pt"
    save_model(model, ckpt_path, formats=config.artifacts.export_formats)

    # Final eval on val.
    final_metrics = _evaluate(model, val_loader, device, config.data.class_names)
    final_metrics["best_val_f1_macro"] = best_val_f1
    final_metrics["checkpoint"] = str(ckpt_path)
    final_metrics["history"] = history

    save_json(final_metrics, Path(config.artifacts.log_dir) / "train_metrics.json")

    _try_end_mlflow(mlflow_run)

    return final_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train weld defect detection model")
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    train(args.config)
