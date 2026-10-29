"""
File:        cqr.py
Description: Conformalized quantile regression: per-segment q_hat on a held-out calibration set ->
             conformal.json.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude

Score of a calibration row is how far the price falls outside [q05, q95] in log space (negative
when inside). The interval widens by the (1 - alpha) conformal quantile of the scores of its
segment, so coverage holds for that segment rather than only on average.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _q_hat(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def fit(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, ptype: np.ndarray, ward: np.ndarray,
        alpha: float, min_rows: int) -> dict:
    """Returns {global, ptype, ward_ptype, alpha}; sparse segments are left out and fall back."""
    scores = np.maximum(lo - y, y - hi)
    df = pd.DataFrame({"s": scores, "p": ptype, "w": ward})
    return {
        "alpha": alpha,
        "min_segment_rows": min_rows,
        "global": _q_hat(scores, alpha),
        "ptype": {p: _q_hat(g["s"].to_numpy(), alpha) for p, g in df.groupby("p") if len(g) >= min_rows},
        "ward_ptype": {
            f"{w}|{p}": _q_hat(g["s"].to_numpy(), alpha)
            for (w, p), g in df.groupby(["w", "p"]) if len(g) >= min_rows
        },
    }


def q_hat_for(conf: dict, ptype: np.ndarray, ward: np.ndarray) -> np.ndarray:
    """Segment offsets per row: (ward, ptype) -> ptype -> global."""
    key = pd.Series([f"{w}|{p}" for w, p in zip(ward, ptype)])
    out = key.map(conf["ward_ptype"])
    out = out.fillna(pd.Series(ptype).map(conf["ptype"]))
    return out.fillna(conf["global"]).to_numpy(dtype="float64")


def apply(conf: dict, lo: np.ndarray, hi: np.ndarray, ptype: np.ndarray, ward: np.ndarray):
    """Widens (lo, hi) log bounds by the segment offset."""
    q = q_hat_for(conf, ptype, ward)
    return lo - q, hi + q
