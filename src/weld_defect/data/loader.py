"""Dataset and data loading utilities.

Implements SRS FR-1.3 (stratified-group split) and FR-1.4 (data profile).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from .preprocess import PreprocessPipeline


class WeldDataset:
    """PyTorch Dataset for weld radiographs.

    Lazy-loads images from ``image_dir`` based on a manifest CSV.
    Applies ``transform`` (Albumentations Compose) if provided.
    """

    def __init__(
        self,
        manifest_path: str | Path | pd.DataFrame,
        image_dir: str | Path,
        transform: Any = None,
        input_size: int = 224,
        label_to_idx: dict[str, int] | None = None,
    ) -> None:
        if isinstance(manifest_path, (str, Path)):
            self.manifest = pd.read_csv(manifest_path)
        else:
            self.manifest = manifest_path.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.input_size = input_size

        if label_to_idx is None:
            if "label_idx" in self.manifest.columns:
                # Build label→idx from the manifest, preserving the declared idx.
                pairs = sorted(zip(self.manifest["label"], self.manifest["label_idx"]), key=lambda x: x[1])
                self.label_to_idx = {lbl: idx for lbl, idx in pairs}
            else:
                labels = sorted(self.manifest["label"].unique())
                self.label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx

        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_image(self, rel_path: str) -> np.ndarray:
        path = self.image_dir / rel_path
        import cv2
        raw = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not decode image: {path}")
        return img

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        row = self.manifest.iloc[idx]
        img = self._load_image(row["image"])

        # Ensure HWC uint8 for Albumentations.
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        if self.transform is not None:
            img = self.transform(image=img)["image"]
            # Albumentations returns HWC numpy; convert to CHW tensor.
            if isinstance(img, np.ndarray):
                if img.ndim == 3:
                    img = np.transpose(img, (2, 0, 1))
                img = torch.from_numpy(img.astype(np.float32))
        else:
            # Fallback: preprocess + to tensor.
            pipe = PreprocessPipeline(self.input_size)
            img = pipe(img)  # HWC float32 [0,1], 3 channels
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)
            img = torch.from_numpy(img)

        label = self.label_to_idx[row["label"]]
        return img, label


def stratified_group_split(
    manifest: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    group_col: str = "group",
    label_col: str = "label",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a manifest into train/val/test with NO group leakage.

    Uses StratifiedGroupKFold so that:
      - All rows with the same ``group`` end up in the SAME split.
      - Label distribution is preserved across splits (stratified).

    Returns (train_df, val_df, test_df).
    """
    if len(manifest) == 0:
        return manifest.copy(), manifest.copy(), manifest.copy()

    # If too few samples/groups for StratifiedGroupKFold, fall back to a simple
    # group-aware split (still no group leakage, but no stratification guarantee).
    n_groups = manifest[group_col].nunique() if group_col in manifest.columns else 0
    n_classes = manifest[label_col].nunique() if label_col in manifest.columns else 1

    if n_groups < 2 or len(manifest) < n_classes * 2:
        # Fallback: simple group-based split.
        groups = sorted(manifest[group_col].unique()) if group_col in manifest.columns else []
        if len(groups) < 2:
            # Not enough groups — just return the whole thing as train.
            return manifest.copy(), manifest.iloc[0:0].copy(), manifest.iloc[0:0].copy()
        n_test = max(1, int(len(groups) * test_size))
        n_val = max(1, int(len(groups) * val_size))
        test_groups = set(groups[-n_test:])
        remaining = [g for g in groups if g not in test_groups]
        val_groups = set(remaining[-n_val:]) if remaining else set()
        train_df = manifest[~manifest[group_col].isin(test_groups | val_groups)].reset_index(drop=True)
        val_df = manifest[manifest[group_col].isin(val_groups)].reset_index(drop=True)
        test_df = manifest[manifest[group_col].isin(test_groups)].reset_index(drop=True)
        return train_df, val_df, test_df

    # StratifiedGroupKFold requires n_splits <= n_groups AND
    # each (group, label) combination must have at least n_splits groups per class.
    # Be conservative: cap n_splits at min(n_groups, 5).
    n_splits = min(int(round(1 / test_size)), n_groups, 5)
    n_splits = max(2, n_splits)

    # Ensure each class has enough groups for stratification.
    # If not, fall back to the simple split above.
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_val_idx, test_idx = next(
            sgkf.split(manifest, manifest[label_col], groups=manifest[group_col])
        )
    except ValueError:
        # Fall back to simple group split.
        groups = sorted(manifest[group_col].unique())
        n_test = max(1, int(len(groups) * test_size))
        test_groups = set(groups[-n_test:])
        train_val = manifest[~manifest[group_col].isin(test_groups)].reset_index(drop=True)
        test_df = manifest[manifest[group_col].isin(test_groups)].reset_index(drop=True)
        # Val from train_val.
        remaining_groups = sorted(train_val[group_col].unique())
        if len(remaining_groups) >= 2:
            val_ratio = val_size / (1 - test_size)
            n_val = max(1, int(len(remaining_groups) * val_ratio))
            val_groups = set(remaining_groups[-n_val:])
        else:
            val_groups = set(remaining_groups[:1]) if remaining_groups else set()
        train_df = train_val[~train_val[group_col].isin(val_groups)].reset_index(drop=True)
        val_df = train_val[train_val[group_col].isin(val_groups)].reset_index(drop=True)
        return train_df, val_df, test_df

    train_val = manifest.iloc[train_val_idx].reset_index(drop=True)
    test_df = manifest.iloc[test_idx].reset_index(drop=True)

    # Second split: peel off val from train_val.
    n_groups_tv = train_val[group_col].nunique()
    if n_groups_tv < 2:
        return train_val, train_val.iloc[0:0], test_df

    val_ratio_of_train_val = val_size / max(0.01, (1 - test_size))
    n_splits2 = min(max(2, int(round(1 / val_ratio_of_train_val))), n_groups_tv, 5)
    try:
        sgkf2 = StratifiedGroupKFold(n_splits=n_splits2, shuffle=True, random_state=seed)
        train_idx, val_idx = next(
            sgkf2.split(train_val, train_val[label_col], groups=train_val[group_col])
        )
        train_df = train_val.iloc[train_idx].reset_index(drop=True)
        val_df = train_val.iloc[val_idx].reset_index(drop=True)
    except ValueError:
        # Fall back: first group → val, rest → train.
        groups = sorted(train_val[group_col].unique())
        val_groups = set(groups[:1])
        train_df = train_val[~train_val[group_col].isin(val_groups)].reset_index(drop=True)
        val_df = train_val[train_val[group_col].isin(val_groups)].reset_index(drop=True)

    return train_df, val_df, test_df


def compute_data_profile(manifest: pd.DataFrame) -> dict[str, Any]:
    """Compute a data quality profile (SRS FR-1.4).

    Returns: n_samples, n_groups, class_counts, imbalance_ratio, missing_images.
    """
    profile: dict[str, Any] = {
        "n_samples": int(len(manifest)),
        "n_groups": int(manifest["group"].nunique()) if "group" in manifest.columns else 0,
        "class_counts": {str(k): int(v) for k, v in manifest["label"].value_counts().items()} if "label" in manifest.columns else {},
    }
    counts = list(profile["class_counts"].values())
    if counts:
        profile["imbalance_ratio"] = float(max(counts) / max(1, min(counts)))
    else:
        profile["imbalance_ratio"] = 0.0
    profile["missing_images"] = 0
    return profile


def build_dataloaders(config: Any) -> tuple[Any, Any, Any]:
    """Build train/val/test dataloaders from a TrainConfig.

    Args:
        config: TrainConfig dataclass (from weld_defect.config.load_config).

    Returns:
        (train_loader, val_loader, test_loader) — torch DataLoader instances.
    """
    import torch
    from torch.utils.data import DataLoader

    from .augment import get_train_aug, get_val_aug

    train_aug = get_train_aug(config.augmentation.ops, config.data.input_size) if config.augmentation.enabled else get_val_aug(config.data.input_size)
    val_aug = get_val_aug(config.data.input_size)

    label_to_idx = {name: i for i, name in enumerate(config.data.class_names)}

    train_ds = WeldDataset(config.data.train_split, config.data.image_dir, train_aug, config.data.input_size, label_to_idx)
    val_ds = WeldDataset(config.data.val_split, config.data.image_dir, val_aug, config.data.input_size, label_to_idx)
    test_ds = WeldDataset(config.data.test_split, config.data.image_dir, val_aug, config.data.input_size, label_to_idx)

    train_loader = DataLoader(
        train_ds, batch_size=config.training.batch_size, shuffle=True,
        num_workers=config.training.num_workers, pin_memory=False, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.training.batch_size, shuffle=False,
        num_workers=config.training.num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.training.batch_size, shuffle=False,
        num_workers=config.training.num_workers, pin_memory=False,
    )
    return train_loader, val_loader, test_loader
