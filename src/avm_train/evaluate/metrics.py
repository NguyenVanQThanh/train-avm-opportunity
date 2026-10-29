"""
File:        metrics.py
Description: Valuation accuracy metrics on price (VND), plus slicing.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude

Point: MdAPE, MAPE, PPE5/10/15/20, bias, RMSLE, MAE(log), R2(log), COD, PRD, PRB.
Interval: coverage, tail misses, relative width, interval score (log space).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PPE_LEVELS = (0.05, 0.10, 0.15, 0.20)
# PRB needs a few points to regress; below this the slope is reported as None.
_MIN_PRB_ROWS = 10


def _num(x) -> float | None:
    """JSON-safe float: NaN/inf become None."""
    x = float(x)
    return x if np.isfinite(x) else None


def point_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    """Point-estimate quality of predicted vs actual prices."""
    ape = np.abs(pred - actual) / actual
    ratio = pred / actual
    med = float(np.median(ratio))
    la, lp = np.log(actual), np.log(pred)
    ss_tot = float(((la - la.mean()) ** 2).sum())
    out = {
        "n": len(actual),
        "mdape": _num(np.median(ape)),
        "mape": _num(ape.mean()),
        # Median signed error: > 0 over-valued, < 0 under-valued.
        "bias": _num(np.median(ratio) - 1),
        "rmsle": _num(np.sqrt(np.mean((lp - la) ** 2))),
        "mae_log": _num(np.mean(np.abs(lp - la))),
        "r2_log": _num(1 - ((lp - la) ** 2).sum() / ss_tot) if ss_tot > 0 else None,
        "cod": _num(100 * np.mean(np.abs(ratio - med)) / med),
        # Mean ratio over value-weighted ratio; > 1.03 means expensive homes are under-valued.
        "prd": _num(ratio.mean() / (pred.sum() / actual.sum())),
        "prb": _prb(actual, pred, ratio, med),
    }
    for lvl in PPE_LEVELS:
        out[f"ppe{round(lvl * 100):02d}"] = _num((ape <= lvl).mean())
    return out


def _prb(actual, pred, ratio, med) -> float | None:
    """IAAO price-related bias: change in relative error per doubling of value (0 = unbiased)."""
    if len(actual) < _MIN_PRB_ROWS:
        return None
    value = 0.5 * (actual + pred / med)
    x = np.log(value / np.median(value)) / np.log(2)
    y = (ratio - med) / med
    return _num(np.polyfit(x, y, 1)[0])


def interval_metrics(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> dict:
    """Interval quality; the score is the Gneiting-Raftery interval score in log price (lower is better)."""
    la, llo, lhi = np.log(actual), np.log(lo), np.log(hi)
    score = (lhi - llo) + (2 / alpha) * np.maximum(llo - la, 0) + (2 / alpha) * np.maximum(la - lhi, 0)
    return {
        "coverage": _num(((actual >= lo) & (actual <= hi)).mean()),
        # Each tail should miss about alpha/2; a lopsided split means the interval is shifted.
        "miss_below": _num((actual < lo).mean()),
        "miss_above": _num((actual > hi).mean()),
        "median_rel_width": _num(np.median((hi - lo) / actual)),
        "interval_score_log": _num(score.mean()),
    }


def coverage(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    """Share of actual prices inside [lo, hi] and the median relative interval width."""
    return {
        "coverage": float(((actual >= lo) & (actual <= hi)).mean()),
        "median_rel_width": float(np.median((hi - lo) / actual)),
    }


def sliced(df: pd.DataFrame, by: tuple[str, ...] = ("_ptype", "_ward_confidence")) -> dict:
    """point_metrics overall and per slice. df needs `_price` (actual) and `_pred` (VND) columns."""
    out = {"global": point_metrics(df["_price"].to_numpy(), df["_pred"].to_numpy())}
    for col in by:
        out[col] = {
            str(k): point_metrics(g["_price"].to_numpy(), g["_pred"].to_numpy())
            for k, g in df.groupby(col)
        }
    return out


def _both(df: pd.DataFrame, alpha: float | None) -> dict:
    out = point_metrics(df["_price"].to_numpy(), df["_pred"].to_numpy())
    if alpha is not None:
        out.update(interval_metrics(df["_price"].to_numpy(), df["_lo"].to_numpy(), df["_hi"].to_numpy(), alpha))
    return out


def breakdown(df: pd.DataFrame, slices: dict[str, pd.Series], alpha: float | None, min_rows: int) -> dict:
    """Full battery overall and per slice value. df needs _price/_pred (+ _lo/_hi when alpha is given)."""
    out = {"global": _both(df, alpha)}
    for name, labels in slices.items():
        out[name] = {
            str(k): _both(g, alpha)
            for k, g in df.groupby(labels.astype(str), sort=True) if len(g) >= min_rows
        }
    return out


def ward_distribution(df: pd.DataFrame, min_rows: int, worst: int = 10) -> dict:
    """How accuracy spreads across wards with enough rows (a good global MdAPE can hide bad wards)."""
    rows = []
    for ward, g in df.groupby("_ward"):
        if len(g) >= min_rows:
            m = point_metrics(g["_price"].to_numpy(), g["_pred"].to_numpy())
            rows.append({"ward": str(ward), "n": m["n"], "mdape": m["mdape"], "bias": m["bias"]})
    if not rows:
        return {"n_wards": 0}
    w = pd.DataFrame(rows)
    return {
        "n_wards": len(w),
        "mdape_p10": _num(w["mdape"].quantile(0.1)),
        "mdape_median": _num(w["mdape"].median()),
        "mdape_p90": _num(w["mdape"].quantile(0.9)),
        "share_mdape_le_15": _num((w["mdape"] <= 0.15).mean()),
        "worst": w.nlargest(worst, "mdape").to_dict("records"),
    }
