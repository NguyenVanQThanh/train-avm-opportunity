"""
File:        research_missing_loc2.py
Description: Research part 2: selection bias of dropping, project->ward purity, and ward mentions in free text
             (sample).

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

import re

import numpy as np
import pandas as pd

from avm_train.extract.crosswalk import norm

raw = pd.read_parquet("data/v0-hf/raw/listings.parquet")
cw = pd.read_csv("data/v0-hf/raw/crosswalk.csv", dtype=str)
legacy = {norm(p) for p in cw.loc[cw["newProvinceCode"] == "79", "provinceShort"].unique()}
s = raw[raw["province_name"].map(norm).isin(legacy)].copy()
ok = s["district_name"].notna() & s["ward_name"].notna()
s["ppm"] = np.log(s["price"] / s["area"])
s = s[s["price"].between(1e8, 5e11) & s["area"].between(5, 5e4)]
ok = ok.reindex(s.index)
print("selection bias (price in [1e8,5e11], area in [5,5e4]):")
for name, grp in (("complete", s[ok]), ("missing", s[~ok])):
    print(f"  {name:9} n={len(grp):>8} median price={grp['price'].median()/1e9:.2f}ty  median area={grp['area'].median():.0f}  "
          f"median log(price/m2)={grp['ppm'].median():.3f}  p10/p90={grp['ppm'].quantile(.1):.2f}/{grp['ppm'].quantile(.9):.2f}")
print("  ptype share complete:", s[ok]["property_type_name"].value_counts(normalize=True).round(3).head(5).to_dict())
print("  ptype share missing :", s[~ok]["property_type_name"].value_counts(normalize=True).round(3).head(5).to_dict())
print("  missing share by ptype:", s.assign(m=~ok).groupby("property_type_name")["m"].mean().round(3).sort_values(ascending=False).head(6).to_dict())

# project -> ward purity among complete rows
full = s[ok & s["project_name"].notna()].copy()
full["pk"] = full["project_name"].map(norm)
top = full.groupby("pk")["ward_name"].agg(lambda x: x.map(norm).value_counts(normalize=True).iloc[0])
sz = full.groupby("pk").size()
print("project->ward purity weighted:", round(float((top * sz).sum() / sz.sum()), 3), "| projects", len(top), "| purity>=0.95:", round(float((top >= .95).mean()), 3))
miss = s[~ok & s["project_name"].notna()].copy()
miss["pk"] = miss["project_name"].map(norm)
rec = miss["pk"].map(top[top >= 0.95])
print("missing rows with project:", len(miss), "| recoverable via pure project:", int(rec.notna().sum()))
