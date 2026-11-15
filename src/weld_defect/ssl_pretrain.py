"""Self-Supervised Pretraining (MAE-style) for radiograph backbones.

Implements a simplified Masked Autoencoder (MAE) for self-supervised pretraining
on unlabeled radiographs. The encoder is then transferred to the supervised
classification task, improving generalization when labeled data is scarce.

This module pretrains a backbone (e.g. ResNet50, EfficientNetV2-S) on unlabeled
images, then saves the pretrained weights for use in supervised fine-tuning.

Reference: He et al., "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from . import CLASS_NAMES
from .data.preprocess import PreprocessPipeline
from .utils.io import ensure_dir, save_json
from .utils.seed import seed_everything


class MAEEncoder(nn.Module):
    """Wrap a timm backbone as a feature encoder for MAE."""

    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = False) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as e:  # pragma: no cover
            raise ImportError("timm is required for MAE pretraining") from e
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.out_channels = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MAEDecoder(nn.Module):
    """Lightweight decoder that reconstructs the masked patches."""

    def __init__(self, encoder_channels: int = 2048, patch_size: int = 16, img_size: int = 224) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2
        self.decoder = nn.Sequential(
            nn.Linear(encoder_channels, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, patch_size * patch_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, encoder_channels) → reconstruct patch pixels
        return self.decoder(features)


class MAE(nn.Module):
    """Masked Autoencoder: encoder + decoder with random patch masking."""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        mask_ratio: float = 0.6,
        patch_size: int = 16,
        img_size: int = 224,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2

        self.encoder = MAEEncoder(backbone_name, pretrained=False)
        self.decoder = MAEDecoder(self.encoder.out_channels, patch_size, img_size)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """Convert (B, C, H, W) → (B, n_patches, patch_size²·C)."""
        p = self.patch_size
        c = images.shape[1]
        h = w = self.img_size
        patches = images.unfold(2, p, p).unfold(3, p, p)  # (B, C, h/p, w/p, p, p)
        patches = patches.contiguous().view(images.shape[0], c, -1, p * p)
        patches = patches.permute(0, 2, 1, 3)  # (B, n_patches, C, p²)
        patches = patches.reshape(images.shape[0], self.n_patches, c * p * p)
        return patches

    def random_masking(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random per-sample masking. Returns (x_masked, mask)."""
        n, _, d = x.shape
        keep = int(self.n_patches * (1 - self.mask_ratio))
        noise = torch.rand(n, self.n_patches, device=x.device)
        ids = torch.argsort(noise, dim=1)
        keep_ids = ids[:, :keep]
        mask = torch.ones(n, self.n_patches, device=x.device)
        mask[:, :keep] = 0
        # Shuffle mask to match ids order
        mask = torch.gather(mask, 1, ids)

        # Keep only visible patches (zero out masked)
        x_masked = x.clone()
        visible = mask.unsqueeze(-1).expand_as(x_masked) == 0
        x_masked = x_masked * visible
        return x_masked, mask

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches = self.patchify(images)  # (B, n_patches, p²·C)
        x_masked, mask = self.random_masking(patches)

        # Encode the full image (simplified — real MAE encodes only visible patches).
        features = self.encoder(images)  # (B, encoder_channels)
        # Broadcast features to all patches for reconstruction.
        feat = features.unsqueeze(1).expand(-1, self.n_patches, -1)  # (B, n_patches, enc_ch)

        # Reconstruct patches — decoder predicts the mean intensity of each patch.
        pred = self.decoder(feat)  # (B, n_patches, p²)
        # Reduce pred to per-patch scalar (mean over patch pixels).
        pred_scalar = pred.mean(dim=-1)  # (B, n_patches)

        # Target: mean intensity of each patch.
        target = patches.mean(dim=-1)  # (B, n_patches)

        # Loss: MSE on masked patches only.
        loss = ((pred_scalar - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)
        return loss, mask

    def get_encoder(self) -> MAEEncoder:
        """Return the pretrained encoder for transfer to classification."""
        return self.encoder


class UnlabeledImageDataset(Dataset):
    """Dataset that loads images without labels (for SSL pretraining)."""

    def __init__(self, image_dir: str | Path, input_size: int = 224) -> None:
        import cv2

        self.image_dir = Path(image_dir)
        self.input_size = input_size
        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
        self.files: list[Path] = []
        for ext in exts:
            self.files.extend(self.image_dir.rglob(ext))
        self.pipeline = PreprocessPipeline(input_size=input_size)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        import cv2

        path = self.files[idx]
        raw = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if img is None:
            # Fallback: return a random tensor
            return torch.zeros(3, self.input_size, self.input_size)
        img = self.pipeline(img)  # HWC float32 [0,1], 3 channels
        img = np.transpose(img, (2, 0, 1))  # CHW
        return torch.from_numpy(img.astype(np.float32))


def pretrain_backbone(
    image_dir: str | Path,
    backbone_name: str = "resnet50",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    mask_ratio: float = 0.6,
    patch_size: int = 16,
    input_size: int = 224,
    output_path: str | Path = "models/mae_pretrained.pt",
    device: str = "cpu",
) -> dict[str, Any]:
    """Run MAE self-supervised pretraining on unlabeled radiographs.

    Args:
        image_dir: Directory of unlabeled images (recursive).
        backbone_name: timm backbone name (e.g. 'resnet50', 'efficientnetv2_s').
        epochs: Pretraining epochs.
        batch_size: Batch size.
        lr: Learning rate.
        mask_ratio: Fraction of patches to mask (0.0-1.0).
        patch_size: Patch size for masking.
        input_size: Image size.
        output_path: Where to save the pretrained encoder weights.
        device: 'cpu' or 'cuda'.

    Returns:
        Dict with pretraining metrics.
    """
    seed_everything(42)
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    dataset = UnlabeledImageDataset(image_dir, input_size=input_size)
    if len(dataset) == 0:
        raise ValueError(f"No images found in {image_dir}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = MAE(
        backbone_name=backbone_name,
        mask_ratio=mask_ratio,
        patch_size=patch_size,
        img_size=input_size,
    ).to(dev)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(dev)
            optimizer.zero_grad()
            loss, _ = model(batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        history.append({"epoch": epoch + 1, "reconstruction_loss": avg_loss})
        print(f"MAE Epoch {epoch+1}/{epochs} | loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.encoder.state_dict().items()}

    # Save pretrained encoder.
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    if best_state is not None:
        torch.save({
            "backbone": backbone_name,
            "state_dict": best_state,
            "pretrain_epochs": epochs,
            "reconstruction_loss": best_loss,
        }, output_path)

    result = {
        "backbone": backbone_name,
        "epochs": epochs,
        "final_loss": history[-1]["reconstruction_loss"] if history else 0.0,
        "best_loss": best_loss,
        "n_images": len(dataset),
        "checkpoint": str(output_path),
        "history": history,
    }
    save_json(result, str(output_path).replace(".pt", "_metrics.json"))
    return result


def load_pretrained_encoder(
    checkpoint_path: str | Path,
    backbone_name: str | None = None,
) -> dict[str, torch.Tensor]:
    """Load pretrained encoder weights for transfer to a classifier.

    Args:
        checkpoint_path: Path to the MAE checkpoint.
        backbone_name: If provided, verify the checkpoint matches.

    Returns:
        State dict of the encoder (compatible with timm backbone).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if backbone_name and ckpt.get("backbone") != backbone_name:
        print(f"Warning: checkpoint backbone ({ckpt.get('backbone')}) != requested ({backbone_name})")
    return ckpt["state_dict"] if "state_dict" in ckpt else ckpt
