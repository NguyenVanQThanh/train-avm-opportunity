"""
File:        rules.py
Description: Row filters on crosswalked listings: bounds, dedup, MAD outliers on log(price/area).

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude

Each rule is a small function returning the filtered frame so run() can report how many rows every
rule removed. Out-of-range frontage/road width become NaN (data-entry errors, not extreme values).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from avm_train.config import stage_dir

# HF already groups property types into these five; anything else is out of scope for v0.
PTYPE_GROUPS = {"Nhà", "Căn hộ chung cư", "Đất", "Biệt thự/Nhà liền kề", "Shophouse"}

# Scales MAD to a standard-deviation equivalent so mad_k reads as "k sigma" for normal data.
_MAD_SCALE = 1.4826


def drop_price_out_of_range(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """Keeps rows whose price is non-null and within [lo, hi]; zero/NaN are dropped."""
    return df[df["price"].between(lo, hi)]


def drop_area_out_of_range(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """Keeps rows whose area_m2 is within [lo, hi]."""
    return df[df["area"].between(lo, hi)]


def keep_ptype_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps the five in-scope property type groups."""
    return df[df["property_type_name"].isin(PTYPE_GROUPS)]


def null_out_of_range(df: pd.DataFrame, col: str, lo: float, hi: float) -> tuple[pd.DataFrame, int]:
    """Sets values outside [lo, hi] to NaN in a copy; returns (frame, number nulled)."""
    out = df.copy()
    bad = out[col].notna() & ~out[col].between(lo, hi)
    out.loc[bad, col] = np.nan
    return out, int(bad.sum())


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drops repeated postings of the same listing, keeping the newest.

    Identity is (ward, street, area, price, publish day): the source has no listing id.
    """
    key = pd.DataFrame(
        {
            "w": df["ward_code_db"],
            "s": df["street_name"].fillna(""),
            "a": df["area"],
            "p": df["price"],
            "d": df["published_at"].dt.normalize(),
        }
    )
    newest_first = df["published_at"].sort_values(ascending=False).index
    return df.loc[newest_first][~key.loc[newest_first].duplicated()].sort_index()


def _robust_stats(x: pd.Series, keys: list[pd.Series]) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Per-group median, MAD and size of x, broadcast back to rows."""
    g = x.groupby(keys)
    med = g.transform("median")
    mad = (x - med).abs().groupby(keys).transform("median")
    return med, mad, g.transform("size")


def drop_mad_outliers(df: pd.DataFrame, k: float, min_rows: int) -> pd.DataFrame:
    """Drops rows whose log(price/area) is more than k scaled-MADs from their cell median.

    The cell is (ward_code_db, ptype_group); cells with fewer than min_rows rows, or a zero MAD,
    use the ptype_group-wide statistics instead.
    """
    x = np.log(df["price"] / df["area"])
    ptype = df["property_type_name"]
    med_c, mad_c, n_c = _robust_stats(x, [df["ward_code_db"], ptype])
    med_g, mad_g, _ = _robust_stats(x, [ptype])
    use_cell = (n_c >= min_rows) & (mad_c > 0)
    med = med_c.where(use_cell, med_g)
    mad = mad_c.where(use_cell, mad_g)
    return df[(x - med).abs() <= k * _MAD_SCALE * mad]


def run(cfg: dict, args=None) -> None:
    """Runs stage `clean`: drops out-of-scope rows and nulls implausible optional values.

    Process:
        1. Log how many prices sit below the sale floor (sale/rent is unlabelled).
        2. Apply the row filters in order, counting rows removed by each.
        3. Null out-of-range frontage and road width instead of dropping the row.
        4. Write listings.parquet and clean_report.json.
    """
    sc, cl = cfg["scope"], cfg["cleaning"]
    df = pd.read_parquet(stage_dir(cfg, "crosswalk") / "listings.parquet")
    out = stage_dir(cfg, "clean")

    report: dict = {"rows_in": len(df)}
    priced = df["price"].dropna()
    # 1. Sale/rent are not labelled in the source; a price floor is the only separator, so log its size.
    report["share_price_below_floor"] = round(float((priced < sc["price_vnd"][0]).mean()), 4)

    # 2. Order matters: MAD runs last so cell medians ignore rows already out of scope.
    steps = [
        ("price_out_of_range", lambda d: drop_price_out_of_range(d, *sc["price_vnd"])),
        ("area_out_of_range", lambda d: drop_area_out_of_range(d, *sc["area_m2"])),
        ("ptype_out_of_scope", keep_ptype_groups),
        ("duplicate", dedupe),
        ("mad_outlier", lambda d: drop_mad_outliers(d, cl["mad_k"], cl["mad_min_cell_rows"])),
    ]
    removed = {}
    for name, fn in steps:
        before = len(df)
        df = fn(df)
        removed[name] = before - len(df)

    # 3. Optional fields are nulled, not row-dropped: the listing price/area are still valid.
    nulled = {}
    for col, key in (("frontage_width", "frontage_m"), ("road_width", "road_width_m")):
        df, nulled[col] = null_out_of_range(df, col, *sc[key])

    # 4. Persist data and report.
    report.update(rows_removed=removed, values_nulled=nulled, rows_out=len(df))
    report["rows_by_ptype"] = df["property_type_name"].value_counts().to_dict()
    report["rows_by_ward_confidence"] = df["ward_confidence"].value_counts().to_dict()
    df.reset_index(drop=True).to_parquet(out / "listings.parquet", index=False)
    (out / "clean_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"clean: {report['rows_in']:,} -> {report['rows_out']:,}  removed={removed}")
