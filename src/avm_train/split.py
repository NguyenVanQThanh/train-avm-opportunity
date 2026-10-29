"""
File:        split.py
Description: Time split (train/val/test_a) + GroupKFold by ward_code_db on train. No random k-fold.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude

Geographic data leaks under random folds (neighbouring listings in both sides), so folds hold out
whole wards and Test A holds out a later month.
"""

from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import GroupKFold

from avm_train.config import GateError, stage_dir

PARTS = ("train", "val", "test_a")


def assign_parts(published_at: pd.Series, ranges: dict) -> pd.Series:
    """Labels each row train/val/test_a by calendar day (inclusive ranges); rows in no range are NaN."""
    day = published_at.dt.normalize()
    label = pd.Series(pd.NA, index=published_at.index, dtype="object")
    for part in PARTS:
        lo, hi = (pd.Timestamp(d) for d in ranges[part])
        label[(day >= lo) & (day <= hi)] = part
    return label


def assign_folds(wards: pd.Series, n_folds: int) -> pd.Series:
    """Fold id 0..n-1 per row such that a ward never spans two folds."""
    folds = pd.Series(-1, index=wards.index, dtype="int64")
    for i, (_, held_out) in enumerate(GroupKFold(n_splits=n_folds).split(wards, groups=wards)):
        folds.iloc[held_out] = i
    return folds


def run(cfg: dict, args=None) -> None:
    """Runs stage `split`: time-based train/val/test_a with ward-grouped folds on train.

    Process:
        1. Label each row by publish day and drop rows outside every range.
        2. Assign ward-grouped folds on train; fail if a ward spans folds.
        3. Build the split report (rows, wards, unseen wards, ptype mix).
        4. Write the three parquet parts and split_report.json.
    """
    sp = cfg["split"]
    df = pd.read_parquet(stage_dir(cfg, "clean") / "listings.parquet")
    out = stage_dir(cfg, "split")

    # 1. Time split; rows between ranges are gaps, not data.
    df["part"] = assign_parts(df["published_at"], sp)
    unassigned = int(df["part"].isna().sum())
    df = df[df["part"].notna()].copy()

    # 2. Ward-grouped folds keep encodings and the unseen-ward check leak-free.
    train = df[df["part"] == "train"].copy()
    train["fold"] = assign_folds(train["ward_code_db"], sp["group_kfold"])
    parts = {"train": train, "val": df[df["part"] == "val"], "test_a": df[df["part"] == "test_a"]}

    ward_folds = train.groupby("ward_code_db")["fold"].nunique()
    if (ward_folds > 1).any() or (train["fold"] < 0).any():
        raise GateError("fold assignment is broken: a ward spans several folds or a row has no fold")

    # 3. Unseen-ward counts tell how much of val/test needs the ward-fallback path.
    train_wards = set(train["ward_code_db"])
    report = {
        "rows_dropped_outside_ranges": unassigned,
        "rows": {k: len(v) for k, v in parts.items()},
        "wards": {k: int(v["ward_code_db"].nunique()) for k, v in parts.items()},
        "wards_unseen_in_train": {
            k: int((~v["ward_code_db"].drop_duplicates().isin(train_wards)).sum())
            for k, v in parts.items() if k != "train"
        },
        "rows_in_unseen_wards": {
            k: int((~v["ward_code_db"].isin(train_wards)).sum()) for k, v in parts.items() if k != "train"
        },
        "fold_rows": train["fold"].value_counts().sort_index().to_dict(),
        "published_at_range": {
            k: [str(v["published_at"].min()), str(v["published_at"].max())] for k, v in parts.items()
        },
        "rows_by_ptype": {k: v["property_type_name"].value_counts().to_dict() for k, v in parts.items()},
    }
    # 4. Persist parts and report.
    for k, v in parts.items():
        v.drop(columns="part").reset_index(drop=True).to_parquet(out / f"{k}.parquet", index=False)
    (out / "split_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=int), encoding="utf-8"
    )
    print(f"split: rows={report['rows']} unseen-ward rows={report['rows_in_unseen_wards']}")
