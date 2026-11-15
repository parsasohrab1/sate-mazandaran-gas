"""Tests for v2 optimization suite: SE attention, EfficientNetV2-S, SAM, TTA, SSL pretraining."""

from __future__ import annotations

import numpy as np
import pytest
import torch

# --- 1. SE Attention Blocks ---

from weld_defect.models.attention import SEBlock, MultiScaleFusion


def test_se_block_forward_shape():
    """SEBlock preserves spatial dimensions, modifies channels."""
    se = SEBlock(channels=64, reduction=16)
    x = torch.randn(2, 64, 32, 32)
    out = se(x)
    assert out.shape == (2, 64, 32, 32)  # same shape as input


def test_se_block_recalibrates():
    """SEBlock output is different from input (attention is applied)."""
    se = SEBlock(channels=32, reduction=8)
    x = torch.randn(1, 32, 16, 16)
    out = se(x)
    assert not torch.allclose(out, x)


def test_se_block_min_reduction():
    """SEBlock with small channels uses minimum reduction of 4."""
    se = SEBlock(channels=8, reduction=16)
    x = torch.randn(1, 8, 4, 4)
    out = se(x)
    assert out.shape == (1, 8, 4, 4)


def test_multiscale_fusion_shape():
    """MultiScaleFusion produces the correct output channels."""
    msf = MultiScaleFusion(in_channels=32, out_channels=64)
    x = torch.randn(2, 32, 32, 32)
    out = msf(x)
    assert out.shape == (2, 64, 32, 32)


# --- 2. CustomCNNv2 with SE attention ---

from weld_defect.models.custom_cnn_v2 import CustomCNNv2


def test_custom_cnn_v2_forward():
    """CustomCNNv2 forward pass produces correct output shape."""
    model = CustomCNNv2(num_classes=6)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 6)


def test_custom_cnn_v2_param_count():
    """CustomCNNv2 has slightly more params than v1 due to SE blocks."""
    model = CustomCNNv2(num_classes=6)
    n = sum(p.numel() for p in model.parameters())
    # SE blocks add ~10-15% params over base CustomCNN (~322K)
    assert 300_000 < n < 700_000, f"Param count {n} outside range"


def test_custom_cnn_v2_has_se_blocks():
    """CustomCNNv2 contains SEBlock modules."""
    model = CustomCNNv2(num_classes=6)
    se_count = sum(1 for m in model.modules() if isinstance(m, SEBlock))
    assert se_count == 3  # one SE block per conv block (3 blocks)


# --- 3. EfficientNetV2-S ---

from weld_defect.models.efficientnet_v2 import EfficientNetV2SClassifier


def test_efficientnetv2_s_forward():
    """EfficientNetV2-S forward pass produces correct output shape."""
    model = EfficientNetV2SClassifier(num_classes=6, pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 6)


def test_efficientnetv2_s_param_count():
    """EfficientNetV2-S has ~20M params (4× more than B0)."""
    model = EfficientNetV2SClassifier(num_classes=6, pretrained=False)
    n = sum(p.numel() for p in model.parameters())
    assert 15_000_000 < n < 25_000_000, f"Param count {n} outside range"


# --- 4. Factory integration ---

from weld_defect.models.factory import build_model


def test_build_model_custom_cnn_v2():
    """build_model('custom_cnn_v2') returns CustomCNNv2."""
    model = build_model("custom_cnn_v2", num_classes=6, pretrained=False)
    assert isinstance(model, CustomCNNv2)


def test_build_model_efficientnetv2_s():
    """build_model('efficientnetv2_s') returns EfficientNetV2SClassifier."""
    model = build_model("efficientnetv2_s", num_classes=6, pretrained=False)
    assert isinstance(model, EfficientNetV2SClassifier)


def test_build_v2_ensemble():
    """build_model('ensemble') with v2 members includes V2 models."""
    model = build_model(
        "ensemble",
        num_classes=6,
        pretrained=False,
        members=["custom_cnn_v2"],
    )
    from weld_defect.models.ensemble import EnsembleModel
    assert isinstance(model, EnsembleModel)
    assert len(model.members) == 1
    assert isinstance(model.members[0], CustomCNNv2)


# --- 5. SAM Optimizer ---

from weld_defect.sam import SAM


def test_sam_initialization():
    """SAM optimizer initializes correctly."""
    model = CustomCNNv2(num_classes=6)
    optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=1e-4, rho=0.05)
    assert optimizer.rho == 0.05
    assert len(optimizer.param_groups) > 0


def test_sam_first_second_step():
    """SAM first_step + second_step updates parameters."""
    model = CustomCNNv2(num_classes=6)
    optimizer = SAM(model.parameters(), torch.optim.SGD, lr=0.01, rho=0.05)
    criterion = torch.nn.CrossEntropyLoss()

    x = torch.randn(4, 3, 64, 64)
    y = torch.tensor([0, 1, 2, 3])

    # Snapshot initial params.
    initial_params = [p.clone() for p in model.parameters()]

    # SAM two-step.
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.first_step(zero_grad=True)
    loss2 = criterion(model(x), y)
    loss2.backward()
    optimizer.second_step(zero_grad=True)

    # Params should have changed.
    changed = any(not torch.allclose(p, ip) for p, ip in zip(model.parameters(), initial_params))
    assert changed


def test_sam_rho_must_be_positive():
    """SAM with non-positive rho raises ValueError."""
    model = CustomCNNv2(num_classes=6)
    with pytest.raises(ValueError):
        SAM(model.parameters(), torch.optim.AdamW, rho=0)


def test_sam_step_raises():
    """SAM.step() raises RuntimeError (must use first_step + second_step)."""
    model = CustomCNNv2(num_classes=6)
    optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=1e-4)
    with pytest.raises(RuntimeError):
        optimizer.step()


# --- 6. TTA (Test-Time Augmentation) in predict.py ---

from weld_defect.predict import WeldPredictor, DEFAULT_TTA_TRANSFORMS


def test_tta_transforms_count():
    """Default TTA has 4 transforms (identity + hflip + rotate ±5)."""
    assert len(DEFAULT_TTA_TRANSFORMS) == 4


def test_predict_with_tta(tiny_checkpoint, sample_image_uint8):
    """predict() with TTA enabled returns valid results."""
    p = WeldPredictor(
        checkpoint_path=str(tiny_checkpoint),
        architecture="custom_cnn",
        input_size=64,
        tta_enabled=True,
    )
    result = p.predict(sample_image_uint8, image_name="test.png")
    assert result["tta_used"] is True
    assert result["predicted_class"] in ["no_defect", "porosity", "crack", "lack_of_fusion", "undercut", "slag_inclusion"]
    assert result["decision_support_only"] is True


def test_predict_without_tta(tiny_checkpoint, sample_image_uint8):
    """predict() without TTA has tta_used=False."""
    p = WeldPredictor(
        checkpoint_path=str(tiny_checkpoint),
        architecture="custom_cnn",
        input_size=64,
        tta_enabled=False,
    )
    result = p.predict(sample_image_uint8)
    assert result["tta_used"] is False


def test_tta_reproducible(tiny_checkpoint, sample_image_uint8):
    """Same image + TTA → same prediction."""
    p = WeldPredictor(
        checkpoint_path=str(tiny_checkpoint),
        architecture="custom_cnn",
        input_size=64,
        tta_enabled=True,
    )
    r1 = p.predict(sample_image_uint8)
    r2 = p.predict(sample_image_uint8)
    assert r1["predicted_class"] == r2["predicted_class"]
    assert abs(r1["confidence"] - r2["confidence"]) < 1e-4


# --- 7. SSL Pretraining (MAE) ---

from weld_defect.ssl_pretrain import MAE, MAEEncoder, MAEDecoder, UnlabeledImageDataset


def test_mae_encoder():
    """MAEEncoder creates a timm backbone."""
    enc = MAEEncoder("resnet18", pretrained=False)
    x = torch.randn(1, 3, 64, 64)
    out = enc(x)
    assert out.dim() == 2  # (B, features)
    assert out.shape[0] == 1


def test_mae_decoder():
    """MAEDecoder reconstructs patches."""
    dec = MAEDecoder(encoder_channels=512, patch_size=16, img_size=64)
    features = torch.randn(2, 512)
    # Decoder expects (B, n_patches, enc_channels)
    feat = features.unsqueeze(1).expand(-1, 16, -1)
    out = dec(feat)
    assert out.shape == (2, 16, 256)  # (B, n_patches, patch_size²)


def test_mae_forward():
    """MAE forward returns loss and mask."""
    mae = MAE(backbone_name="resnet18", mask_ratio=0.5, patch_size=16, img_size=64)
    x = torch.randn(2, 3, 64, 64)
    loss, mask = mae(x)
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0
    assert mask.shape == (2, 16)  # (B, n_patches)


def test_mae_mask_ratio():
    """MAE masks approximately mask_ratio fraction of patches."""
    mae = MAE(backbone_name="resnet18", mask_ratio=0.6, patch_size=16, img_size=64)
    x = torch.randn(4, 3, 64, 64)
    _, mask = mae(x)
    # ~60% should be masked (value=1)
    masked_frac = mask.mean().item()
    assert 0.4 < masked_frac < 0.8  # allow variance due to random masking


def test_unlabeled_dataset(tmp_path, sample_image_uint8):
    """UnlabeledImageDataset loads images from a directory."""
    import cv2
    for i in range(5):
        cv2.imwrite(str(tmp_path / f"img_{i}.png"), sample_image_uint8)

    ds = UnlabeledImageDataset(tmp_path, input_size=64)
    assert len(ds) == 5
    img = ds[0]
    assert img.shape == (3, 64, 64)


@pytest.mark.slow
def test_pretrain_backbone_smoke(tmp_path, sample_image_uint8):
    """pretrain_backbone runs a 1-epoch MAE pretraining on a few images."""
    import cv2
    for i in range(6):
        cv2.imwrite(str(tmp_path / f"img_{i}.png"), sample_image_uint8)

    from weld_defect.ssl_pretrain import pretrain_backbone
    result = pretrain_backbone(
        image_dir=tmp_path,
        backbone_name="resnet18",
        epochs=1,
        batch_size=3,
        lr=1e-4,
        mask_ratio=0.5,
        patch_size=16,
        input_size=64,
        output_path=tmp_path / "mae.pt",
        device="cpu",
    )
    assert result["n_images"] == 6
    assert result["epochs"] == 1
    assert (tmp_path / "mae.pt").exists()
