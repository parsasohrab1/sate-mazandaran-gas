"""Dataset and dataloader construction.

Splits are stored as CSV with columns: ``image``, ``label``, ``group``.
``group`` is the pipeline segment / weld-seam id — used to prevent leakage
across train/val/test (the same weld appearing in two splits).

The pure-pandas split function works without torch. The ``WeldDataset`` and
``build_dataloaders`` require torch (imported lazily).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from weld_defect.data.preprocess import PreprocessPipeline
from weld_defect.data.augment import Augmenter


# --------------------------------------------------------------------------- #
#  Splitting (no torch required)
# --------------------------------------------------------------------------- #
def stratified_group_split(
    df: pd.DataFrame,
    label_col: str = "label",
    group_col: str = "group",
    test_size: float = 0.2,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split that never puts the same ``group`` in two splits.

    Falls back to scikit-learn's ``StratifiedShuffleSplit`` over groups when
    available; otherwise uses a deterministic numpy implementation.
    """
    rng = np.random.default_rng(seed)
    groups = df[group_col].astype(str).to_numpy()
    labels = df[label_col].to_numpy()

    # Build a group -> (majority label, size) table to stratify on.
    gdf = df.assign(_g=groups).groupby("_g", sort=False)
    group_labels = gdf[label_col].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    group_names = group_labels.index.to_numpy()
    group_strat = group_labels.to_numpy()

    try:
        from sklearn.model_selection import StratifiedShuffleSplit

        # First cut: trainval vs test
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval_idx, test_idx = next(sss1.split(group_names, group_strat))
        # Second cut: train vs val (relative to trainval)
        rel_val = val_size / (1.0 - test_size)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
        rel_train, rel_val = next(sss2.split(group_names[trainval_idx], group_strat[trainval_idx]))
        train_idx = trainval_idx[rel_train]
        val_idx = trainval_idx[rel_val]
    except (ImportError, ValueError):
        # Fallback: deterministic numpy shuffle (used when sklearn is missing or
        # when a class has too few groups for stratified splitting).
        order = rng.permutation(len(group_names))
        n_test = max(1, int(round(len(order) * test_size)))
        n_val = max(1, int(round(len(order) * val_size)))
        test_idx = order[:n_test]
        val_idx = order[n_test : n_test + n_val]
        train_idx = order[n_test + n_val :]

    def _select(idxs: np.ndarray) -> pd.DataFrame:
        chosen = set(group_names[idxs])
        return df[df[group_col].astype(str).isin(chosen)].reset_index(drop=True)

    return _select(train_idx), _select(val_idx), _select(test_idx)


def write_splits(
    df: pd.DataFrame,
    out_dir: str | Path,
    label_col: str = "label",
    group_col: str = "group",
    **kwargs,
) -> dict[str, Path]:
    """Split ``df`` and write train/val/test CSVs into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr, va, te = stratified_group_split(df, label_col, group_col, **kwargs)
    paths = {
        "train": out / "train.csv",
        "val": out / "val.csv",
        "test": out / "test.csv",
    }
    tr.to_csv(paths["train"], index=False)
    va.to_csv(paths["val"], index=False)
    te.to_csv(paths["test"], index=False)
    return paths


# --------------------------------------------------------------------------- #
#  Dataset / DataLoaders (torch required)
# --------------------------------------------------------------------------- #
class WeldDataset:
    """torch-style dataset over a split CSV.

    Yields ``(image_tensor, label_int)``. Designed to be small and explicit so
    it is easy to test; production loaders may swap in a streaming variant.
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path,
        class_names: Sequence[str],
        preprocess: PreprocessPipeline,
        augment: Augmenter | None = None,
    ) -> None:
        import torch  # noqa: F401  (fail fast if torch missing)
        self.df = pd.read_csv(csv_path)
        if "image" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(f"CSV {csv_path} must have 'image' and 'label' columns")
        self.image_dir = Path(image_dir)
        self.class_names = list(class_names)
        self.label_to_idx = {n: i for i, n in enumerate(self.class_names)}
        self.preprocess = preprocess
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def _load_raw(self, row: pd.Series) -> np.ndarray:
        from weld_defect.data.preprocess import load_image_gray
        p = self.image_dir / str(row["image"])
        if not p.exists():
            # try with extension
            p = self.image_dir / f"{row['image']}.png"
        return load_image_gray(p)

    def __getitem__(self, idx: int):
        import torch
        row = self.df.iloc[idx]
        raw = self._load_raw(row)
        if self.augment is not None:
            raw = self.augment(raw)
        arr = self.preprocess(raw)
        # arr is HWC float; convert to CHW tensor
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        t = torch.from_numpy(arr.transpose(2, 0, 1)).float()
        label_name = row["label"]
        label = self.label_to_idx.get(str(label_name), int(label_name) if str(label_name).lstrip("-").isdigit() else 0)
        return t, int(label)


def build_dataloaders(cfg, num_workers: int | None = None):
    """Build train/val/test torch DataLoaders from a ``TrainConfig``.

    Requires torch and (optionally) albumentations.
    """
    import torch
    from torch.utils.data import DataLoader

    from weld_defect.data.augment import build_augmenter

    pre = PreprocessPipeline(input_size=cfg.data.input_size)
    train_aug = build_augmenter(cfg.augmentation)
    val_aug = None  # no augmentation on val/test

    def _make(csv, aug, shuffle):
        ds = WeldDataset(
            csv_path=csv,
            image_dir=cfg.data.image_dir,
            class_names=cfg.data.class_names,
            preprocess=pre,
            augment=aug,
        )
        return DataLoader(
            ds,
            batch_size=cfg.training.batch_size,
            shuffle=shuffle,
            num_workers=num_workers if num_workers is not None else cfg.training.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=shuffle,
        )

    return (
        _make(cfg.data.train_split, train_aug, shuffle=True),
        _make(cfg.data.val_split, val_aug, shuffle=False),
        _make(cfg.data.test_split, val_aug, shuffle=False),
    )
