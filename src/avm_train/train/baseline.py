"""
File:        baseline.py
Description: Baseline to beat: the ward_price_ref x area estimate (median price/area of the (ward, ptype) cell).

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from avm_train.config import stage_dir
from avm_train.evaluate import metrics


def predict(df: pd.DataFrame) -> np.ndarray:
    """Price estimate = exp(ward_price_ref) * area; ward_price_ref is already train-only for val/test."""
    return np.exp(df["ward_price_ref"].to_numpy() + np.log(df["area_m2"].to_numpy()))


def run(cfg: dict, args=None) -> None:
    """Runs stage `baseline`: scores the ward_price_ref x area estimate the model must beat.

    Process:
        1. Predict val and test_a with the baseline.
        2. Write baseline_metrics.json and print a summary.
    """
    src, out = stage_dir(cfg, "features"), stage_dir(cfg, "baseline")
    report = {}
    # 1. Same metric battery as the model so the comparison is like for like.
    for part in ("val", "test_a"):
        df = pd.read_parquet(src / f"{part}.parquet")
        df["_pred"] = predict(df)
        report[part] = metrics.sliced(df)
    # 2. Persist and print.
    (out / "baseline_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for part, r in report.items():
        g = r["global"]
        print(f"baseline {part}: n={g['n']:,} MdAPE={g['mdape']:.3f} PPE15={g['ppe15']:.3f}")
