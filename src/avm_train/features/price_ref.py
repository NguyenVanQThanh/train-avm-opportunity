"""
File:        price_ref.py
Description: ward_price_ref / street_price_ref target encodings.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude

ward_price_ref: median log(price/area) of the (ward, ptype) cell; cells under min_rows fall back to
the ptype-wide median, then to the global median.
street_price_ref: ward_price_ref plus the street's mean residual (log ppm minus ward ref), shrunk
toward zero with weight n / (n + m), so a street seen once barely moves the estimate.

Training rows use out-of-fold tables (see oof) so a row never contributes to its own encoding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from avm_train.features.normalize import street_key


@dataclass(frozen=True)
class Tables:
    cell: pd.Series      # (ward, ptype) -> median log ppm, only cells with >= min_rows
    ptype: pd.Series     # ptype -> median log ppm
    glob: float
    street: pd.DataFrame  # (ward, street) -> n, mean_resid
    min_rows: int
    m: float


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Reduces a listing frame to the encoding keys (ward, ptype, street) and log ppm if priced."""
    out = pd.DataFrame(
        {
            "ward": df["ward_code_db"].to_numpy(),
            "ptype": df["property_type_name"].to_numpy(),
            "street": df["street_name"].map(street_key).to_numpy() if "street_name" in df else None,
        }
    )
    if {"price", "area"} <= set(df.columns):
        out["ppm"] = np.log(df["price"].to_numpy() / df["area"].to_numpy())
    return out


def _ward_ref(base: pd.DataFrame, t: Tables) -> np.ndarray:
    idx = pd.MultiIndex.from_arrays([base["ward"], base["ptype"]])
    v = t.cell.reindex(idx).to_numpy(dtype="float64")
    v = np.where(np.isnan(v), base["ptype"].map(t.ptype).to_numpy(dtype="float64"), v)
    return np.where(np.isnan(v), t.glob, v)


def fit(base: pd.DataFrame, min_rows: int, m: float) -> Tables:
    """Builds the lookup tables from labelled rows only."""
    glob = float(base["ppm"].median())
    ptype = base.groupby("ptype")["ppm"].median()
    g = base.groupby(["ward", "ptype"])["ppm"]
    med, cnt = g.median(), g.size()
    cell = med[cnt >= min_rows]
    t0 = Tables(cell, ptype, glob, pd.DataFrame(columns=["n", "mean_resid"]), min_rows, m)
    resid = pd.Series(base["ppm"].to_numpy() - _ward_ref(base, t0), index=base.index)
    sg = resid.groupby([base["ward"], base["street"]])
    street = pd.DataFrame({"n": sg.size(), "mean_resid": sg.mean()})
    return Tables(cell, ptype, glob, street, min_rows, m)


def apply(base: pd.DataFrame, t: Tables) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ward_price_ref, street_price_ref) for rows; never reads the label."""
    ward = _ward_ref(base, t)
    idx = pd.MultiIndex.from_arrays([base["ward"], base["street"]])
    st = t.street.reindex(idx)
    n = st["n"].fillna(0).to_numpy(dtype="float64")
    mean = st["mean_resid"].fillna(0).to_numpy(dtype="float64")
    return ward, ward + n * mean / (n + t.m)


def oof(base: pd.DataFrame, enc_fold: np.ndarray, min_rows: int, m: float) -> tuple[np.ndarray, np.ndarray]:
    """Encodes each row with tables fit on the other folds only."""
    ward = np.empty(len(base))
    street = np.empty(len(base))
    for f in np.unique(enc_fold):
        held = enc_fold == f
        t = fit(base[~held], min_rows, m)
        ward[held], street[held] = apply(base[held], t)
    return ward, street


def from_json(d: dict) -> Tables:
    """Inverse of to_json: rebuilds the lookup tables exactly as shipped in feature_schema.json."""
    cell_idx = pd.MultiIndex.from_tuples([tuple(k.split("|", 1)) for k in d["ward_ptype"]])
    street_idx = pd.MultiIndex.from_tuples([tuple(k.split("|", 1)) for k in d["street"]])
    street = pd.DataFrame(
        {"n": [v[0] for v in d["street"].values()], "mean_resid": [v[1] for v in d["street"].values()]},
        index=street_idx,
    )
    return Tables(
        pd.Series(list(d["ward_ptype"].values()), index=cell_idx, dtype="float64"),
        pd.Series(d["ptype"], dtype="float64"),
        d["global"], street, d["min_rows"], d["smoothing_m"],
    )


def to_json(t: Tables) -> dict:
    """Serialisable form shipped in feature_schema.json so inference uses the exact same tables."""
    return {
        "global": t.glob,
        "min_rows": t.min_rows,
        "smoothing_m": t.m,
        "ptype": t.ptype.to_dict(),
        "ward_ptype": {f"{w}|{p}": v for (w, p), v in t.cell.items()},
        "street": {f"{w}|{s}": [int(r.n), r.mean_resid] for (w, s), r in t.street.iterrows()},
    }
