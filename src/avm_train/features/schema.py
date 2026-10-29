"""
File:        schema.py
Description: Feature matrix + feature_schema.json.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude

Stage `features`: reads split parquets, writes features/{train,val,test_a}.parquet with the model
columns FEATURES, label `y` = log(price), and underscore-prefixed metadata columns used only for
splitting and metric slicing. Also writes feature_schema.json (order, category code tables and the
final price_ref tables).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from avm_train import config
from avm_train.config import stage_dir
from avm_train.features import price_ref
from avm_train.features.normalize import (
    DIRECTIONS,
    PTYPES,
    code_table,
    encode,
    normalize_direction,
)

FEATURE_VERSION = "fs-hf-0"
# Model input order. Categoricals hold integer codes (float64, NaN = unknown/missing).
FEATURES = [
    "area_m2", "ptype_group", "bedrooms", "bathrooms", "floors", "frontage_m", "road_width_m",
    "house_direction", "has_project", "ward_code_db", "ward_price_ref", "street_price_ref",
]
CATEGORICAL = ["ptype_group", "house_direction", "ward_code_db"]


def build_matrix(df: pd.DataFrame, ward_ref, street_ref, codes: dict) -> pd.DataFrame:
    """Assembles model columns; categories absent from `codes` become NaN rather than a new code."""
    direction = df["house_direction"].map(normalize_direction)
    x = pd.DataFrame(
        {
            "area_m2": df["area"].to_numpy(dtype="float64"),
            "ptype_group": encode(df["property_type_name"], codes["ptype_group"]),
            "bedrooms": df["bedroom_count"].to_numpy(dtype="float64"),
            "bathrooms": df["bathroom_count"].to_numpy(dtype="float64"),
            "floors": df["floor_count"].to_numpy(dtype="float64"),
            "frontage_m": df["frontage_width"].to_numpy(dtype="float64"),
            "road_width_m": df["road_width"].to_numpy(dtype="float64"),
            "house_direction": encode(direction, codes["house_direction"]),
            "has_project": df["project_name"].notna().astype("float64").to_numpy(),
            "ward_code_db": encode(df["ward_code_db"], codes["ward_code_db"]),
            "ward_price_ref": ward_ref,
            "street_price_ref": street_ref,
        },
        index=df.index,
    )
    return x[FEATURES]


def with_meta(x: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    out["y"] = np.log(df["price"].to_numpy(dtype="float64"))
    out["_price"] = df["price"].to_numpy()
    out["_ward"] = df["ward_code_db"].to_numpy()
    out["_ptype"] = df["property_type_name"].to_numpy()
    out["_ward_confidence"] = df["ward_confidence"].to_numpy()
    out["_published_at"] = df["published_at"].to_numpy()
    return out


def make_features(train: pd.DataFrame, others: dict[str, pd.DataFrame], cfg: dict):
    """Builds model matrices: train encoded out-of-fold, `others` encoded with tables fit on all train.

    Returns (train_matrix, {name: matrix}, final_tables, category_codes). Used by the features stage
    and by evaluation, which refits everything on a subset of train for the unseen-ward check.

    Process:
        1. Build category code tables from train.
        2. Encode train price refs out-of-fold and fit the final tables on all train.
        3. Assemble the train matrix.
        4. Encode every other part with the final tables.
    """
    train = train.reset_index(drop=True)
    ft = cfg["features"]
    min_rows, m = ft["price_ref_min_rows"], float(ft["price_ref_smoothing_m"])
    # 1. Codes come from train only; unseen categories in later parts become NaN.
    codes = {
        "ptype_group": code_table(PTYPES),
        "house_direction": code_table(DIRECTIONS),
        "ward_code_db": code_table(sorted(train["ward_code_db"].unique())),
    }

    # 2. Train rows are encoded from the other row-level folds so no row sees its own label.
    # Folds by ward (column `fold`) are reserved for unseen-ward evaluation, where held-out wards
    # have no table cells by construction and would train the model on a feature it never gets.
    base = price_ref.prepare(train)
    enc_fold = np.random.default_rng(cfg["seed"]).integers(0, ft["enc_folds"], len(train))
    w, s = price_ref.oof(base, enc_fold, min_rows, m)
    final = price_ref.fit(base, min_rows, m)

    # 3. Train matrix.
    res = with_meta(build_matrix(train, w, s, codes), train)
    res["_enc_fold"] = enc_fold

    # 4. Val/test use tables fit on all train, exactly what inference will use.
    out = {}
    for k, df in others.items():
        df = df.reset_index(drop=True)
        w, s = price_ref.apply(price_ref.prepare(df), final)
        out[k] = with_meta(build_matrix(df, w, s, codes), df)
    return res, out, final, codes


def run(cfg: dict, args=None) -> None:
    """Runs stage `features`: model matrices plus the feature schema shipped with the model.

    Process:
        1. Build train/val/test_a matrices.
        2. Write the matrices.
        3. Write feature_schema.json with codes, encoding tables and the config hash.
    """
    src, out = stage_dir(cfg, "split"), stage_dir(cfg, "features")
    parts = {k: pd.read_parquet(src / f"{k}.parquet") for k in ("train", "val", "test_a")}
    train = parts["train"].reset_index(drop=True)
    # 1. Matrices.
    res, others, final, codes = make_features(
        train, {k: parts[k] for k in ("val", "test_a")}, cfg)
    # 2. Ward folds ride along for the unseen-ward evaluation.
    res["_fold"] = train["fold"].to_numpy()
    res.to_parquet(out / "train.parquet", index=False)
    for k, df in others.items():
        df.to_parquet(out / f"{k}.parquet", index=False)

    # 3. Inference must reuse these exact tables, so they are shipped, not recomputed.
    ft = cfg["features"]
    schema = {
        "feature_version": FEATURE_VERSION,
        "features": FEATURES,
        "categorical": CATEGORICAL,
        "label": "log(price_vnd)",
        "category_codes": codes,
        "price_ref": price_ref.to_json(final),
        "encoding": {"enc_folds": ft["enc_folds"], "seed": cfg["seed"], "config_sha256": config.sha256(cfg)},
    }
    (out / "feature_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False), encoding="utf-8"
    )
    nan = {c: round(float(res[c].isna().mean()), 3) for c in FEATURES}
    print(f"features: train={len(res):,}; NaN share={nan}")
