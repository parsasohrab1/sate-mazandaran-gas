"""Tests for data.loader."""

from __future__ import annotations

import pandas as pd
import pytest

from weld_defect.data.loader import (
    WeldDataset,
    compute_data_profile,
    stratified_group_split,
)


def test_stratified_group_split_no_leakage(manifest_df):
    """No group appears in more than one split."""
    train, val, test = stratified_group_split(manifest_df, test_size=0.2, val_size=0.1, seed=42)
    train_groups = set(train["group"])
    val_groups = set(val["group"])
    test_groups = set(test["group"])
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_stratified_group_split_preserves_total(manifest_df):
    """train + val + test == original."""
    train, val, test = stratified_group_split(manifest_df, test_size=0.2, val_size=0.1, seed=42)
    assert len(train) + len(val) + len(test) == len(manifest_df)


def test_stratified_group_split_stratification(manifest_df):
    """All classes appear in train split (with enough data)."""
    train, val, test = stratified_group_split(manifest_df, test_size=0.2, val_size=0.1, seed=42)
    assert set(train["label"]) == set(manifest_df["label"])


def test_stratified_group_split_empty():
    """Empty manifest returns empty splits."""
    empty = pd.DataFrame(columns=["image", "label", "label_idx", "group"])
    train, val, test = stratified_group_split(empty)
    assert len(train) == 0
    assert len(val) == 0
    assert len(test) == 0


def test_compute_data_profile(manifest_df):
    """Profile has expected fields."""
    profile = compute_data_profile(manifest_df)
    assert profile["n_samples"] == len(manifest_df)
    assert profile["n_groups"] == 6
    assert "class_counts" in profile
    assert "imbalance_ratio" in profile
    assert profile["imbalance_ratio"] == 1.0  # balanced


def test_weld_dataset_len(tmp_synthetic_dataset):
    """WeldDataset.__len__ matches manifest."""
    manifest = pd.read_csv(tmp_synthetic_dataset / "manifest.csv")
    ds = WeldDataset(manifest, tmp_synthetic_dataset / "images")
    assert len(ds) == len(manifest)


def test_weld_dataset_getitem(tmp_synthetic_dataset):
    """WeldDataset.__getitem__ returns (image, label_idx)."""
    manifest = pd.read_csv(tmp_synthetic_dataset / "manifest.csv")
    ds = WeldDataset(manifest, tmp_synthetic_dataset / "images")
    img, label = ds[0]
    assert label is not None
    assert isinstance(label, int)
    assert 0 <= label < 6
