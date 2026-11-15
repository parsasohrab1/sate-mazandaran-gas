"""Data split tests — pandas + sklearn only (no torch)."""
from __future__ import annotations

import pandas as pd

from weld_defect.data.loader import stratified_group_split, write_splits


def test_split_disjoint_groups(small_manifest_df):
    tr, va, te = stratified_group_split(small_manifest_df, label_col="label_idx",
                                        group_col="group", test_size=0.2, val_size=0.15, seed=42)
    # No group appears in two splits
    g_tr = set(tr["group"])
    g_va = set(va["group"])
    g_te = set(te["group"])
    assert g_tr.isdisjoint(g_va)
    assert g_tr.isdisjoint(g_te)
    assert g_va.isdisjoint(g_te)
    # Union covers all groups
    assert g_tr | g_va | g_te == set(small_manifest_df["group"])
    # Splits cover all rows exactly once
    assert len(tr) + len(va) + len(te) == len(small_manifest_df)


def test_split_sizes_reasonable(small_manifest_df):
    tr, va, te = stratified_group_split(small_manifest_df, label_col="label_idx",
                                        group_col="group", test_size=0.34, val_size=0.33, seed=1)
    n = len(small_manifest_df)
    assert len(tr) > 0 and len(va) > 0 and len(te) > 0
    # With only 6 groups, ceil-rounding can be coarse; use tolerant bounds.
    assert len(tr) / n > 0.1
    assert len(te) / n < 0.7


def test_write_splits_creates_csvs(tmp_path, small_manifest_df):
    paths = write_splits(small_manifest_df, tmp_path, label_col="label_idx", group_col="group",
                         test_size=0.34, val_size=0.33, seed=2)
    assert all(Path.exists(v) for v in paths.values())
    for name, p in paths.items():
        df = pd.read_csv(p)
        assert set(df.columns) >= {"image", "label", "label_idx", "group"}
        assert len(df) > 0


def test_split_deterministic(small_manifest_df):
    a = stratified_group_split(small_manifest_df, label_col="label_idx", group_col="group", seed=7)
    b = stratified_group_split(small_manifest_df, label_col="label_idx", group_col="group", seed=7)
    assert [len(x) for x in a] == [len(x) for x in b]
    assert set(a[0]["group"]) == set(b[0]["group"])


from pathlib import Path  # noqa: E402  (used above)
