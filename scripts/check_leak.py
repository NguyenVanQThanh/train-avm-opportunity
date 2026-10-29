"""
File:        check_leak.py
Description: Sanity: OOF price_ref on train should look like price_ref on val (no leakage), and beat constant.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

import numpy as np
import pandas as pd

tr = pd.read_parquet("data/v0-hf/features/train.parquet")
va = pd.read_parquet("data/v0-hf/features/val.parquet")
for name, d in (("train(OOF)", tr), ("val", va)):
    ppm = d["y"] - np.log(d["area_m2"])
    for c in ("ward_price_ref", "street_price_ref"):
        r = np.corrcoef(d[c], ppm)[0, 1]
        mae = np.abs(d[c] - ppm).mean()
        print(f"{name:11} {c:17} corr={r:.3f} MAE(log ppm)={mae:.3f}")
    print(f"{name:11} constant-median MAE={np.abs(ppm - ppm.median()).mean():.3f}")
# in-sample (leaky) reference: encode train by tables fit on all of train -> would look too good
print("val/train street_price_ref NaN-free:", va["street_price_ref"].notna().all(), tr["street_price_ref"].notna().all())
