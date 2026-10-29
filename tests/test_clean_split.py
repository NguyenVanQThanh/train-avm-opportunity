"""
File:        test_clean_split.py
Description: Tests for cleaning rules and the time/ward split.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

import numpy as np
import pandas as pd

from avm_train import split
from avm_train.clean import rules


def _df(**over):
    base = {
        "price": [5e9] * 4,
        "area": [50.0] * 4,
        "property_type_name": ["Nhà"] * 4,
        "ward_code_db": ["w1"] * 4,
        "street_name": ["A"] * 4,
        "published_at": pd.to_datetime(["2025-07-01"] * 4),
        "frontage_width": [4.0] * 4,
        "road_width": [6.0] * 4,
    }
    base.update(over)
    return pd.DataFrame(base)


def test_price_bounds_drop_zero_nan_and_extremes():
    df = _df(price=[0, np.nan, 5e9, 9e14])
    assert rules.drop_price_out_of_range(df, 1e8, 5e11)["price"].tolist() == [5e9]


def test_area_bounds():
    df = _df(area=[1.0, 5.0, 50.0, 6e7])
    assert rules.drop_area_out_of_range(df, 5, 5e4)["area"].tolist() == [5.0, 50.0]


def test_ptype_out_of_scope_dropped():
    df = _df(property_type_name=["Nhà", "Kho, nhà xưởng", "Đất", "Khác"])
    assert rules.keep_ptype_groups(df)["property_type_name"].tolist() == ["Nhà", "Đất"]


def test_out_of_range_values_become_nan_not_clamped():
    df = _df(frontage_width=[4.0, 9080000.0, 0.2, np.nan])
    out, n = rules.null_out_of_range(df, "frontage_width", 1, 50)
    assert n == 2
    assert out["frontage_width"].tolist()[0] == 4.0
    assert out["frontage_width"].isna().tolist() == [False, True, True, True]
    assert df["frontage_width"].iloc[1] == 9080000.0  # input not mutated


def test_dedupe_keeps_newest_of_same_listing():
    df = _df(published_at=pd.to_datetime(["2025-07-01 08:00", "2025-07-01 20:00", "2025-07-02 00:00", "2025-07-01 09:00"]))
    out = rules.dedupe(df)
    assert len(out) == 2
    assert pd.Timestamp("2025-07-01 20:00") in out["published_at"].tolist()
    assert pd.Timestamp("2025-07-01 08:00") not in out["published_at"].tolist()


def test_mad_drops_outlier_in_dense_cell_and_uses_fallback_in_sparse_cell():
    rng = np.random.default_rng(0)
    n = 40
    dense = pd.DataFrame(
        {
            "price": 1e8 * 50 * np.exp(rng.normal(0, 0.1, n)),
            "area": 50.0,
            "property_type_name": "Nhà",
            "ward_code_db": "dense",
        }
    )
    dense.loc[0, "price"] = dense["price"].median() * 100  # obvious outlier
    sparse = pd.DataFrame(
        {"price": [5e9], "area": [50.0], "property_type_name": "Nhà", "ward_code_db": "sparse"}
    )
    out = rules.drop_mad_outliers(pd.concat([dense, sparse], ignore_index=True), k=4, min_rows=20)
    assert 0 not in out.index
    assert "sparse" in out["ward_code_db"].tolist()
    assert len(out) == n  # 39 dense + 1 sparse


def test_assign_parts_by_inclusive_day():
    ranges = {"train": ["2025-06-01", "2026-01-31"], "val": ["2026-02-01", "2026-02-28"], "test_a": ["2026-03-01", "2026-03-31"]}
    ts = pd.Series(pd.to_datetime(["2026-01-31 23:59", "2026-02-01 00:00", "2026-03-31 23:00", "2026-04-01 00:00", "2025-05-31 00:00"]))
    assert split.assign_parts(ts, ranges).tolist() == ["train", "val", "test_a", pd.NA, pd.NA]


def test_folds_never_split_a_ward_and_use_every_fold():
    wards = pd.Series([f"w{i % 23}" for i in range(500)])
    folds = split.assign_folds(wards, 5)
    assert folds.groupby(wards).nunique().max() == 1
    assert sorted(folds.unique()) == [0, 1, 2, 3, 4]


def test_run_splits_end_to_end(tmp_path, monkeypatch):
    from avm_train import config

    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    cfg = {"run_id": "t", "split": {
        "train": ["2025-06-01", "2026-01-31"], "val": ["2026-02-01", "2026-02-28"],
        "test_a": ["2026-03-01", "2026-03-31"], "group_kfold": 5}}
    dates = ["2025-07-10"] * 60 + ["2026-02-10"] * 5 + ["2026-03-10"] * 5 + ["2026-04-10"] * 3
    wards = [f"w{i % 10}" for i in range(60)] + ["w1"] * 5 + ["w2", "w2", "w2", "new", "new"] + ["w1"] * 3
    clean = tmp_path / "t" / "clean"
    clean.mkdir(parents=True)
    pd.DataFrame({"ward_code_db": wards, "published_at": pd.to_datetime(dates),
                  "property_type_name": "Nhà"}).to_parquet(clean / "listings.parquet")
    split.run(cfg)
    out = tmp_path / "t" / "split"
    train = pd.read_parquet(out / "train.parquet")
    assert len(train) == 60 and "fold" in train
    assert len(pd.read_parquet(out / "val.parquet")) == 5
    test_a = pd.read_parquet(out / "test_a.parquet")
    assert len(test_a) == 5
    rep = __import__("json").loads((out / "split_report.json").read_text(encoding="utf-8"))
    assert rep["rows_dropped_outside_ranges"] == 3
    assert rep["rows_in_unseen_wards"]["test_a"] == 2
