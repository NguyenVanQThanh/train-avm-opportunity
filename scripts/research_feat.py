"""
File:        research_feat.py
Description: Distribution check for feature normalizers. Read-only.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

import pandas as pd

from avm_train.extract.crosswalk import norm

df = pd.read_parquet("data/v0-hf/split/train.parquet")
print("house_direction raw:", df["house_direction"].value_counts(dropna=False).to_dict())
st = df["street_name"].dropna()
k = st.map(norm)
print("street nonnull share:", round(len(st) / len(df), 3), "| distinct raw:", st.nunique(), "| distinct norm:", k.nunique())
print("first-word share:", k.str.split().str[0].value_counts(normalize=True).head(12).round(3).to_dict())
key = pd.DataFrame({"w": df["ward_code_db"], "k": df["street_name"].map(norm)}).dropna()
sz = key.groupby(["w", "k"]).size()
print("(ward,street) keys:", len(sz), "| rows covered by keys n>=5:", round(float(sz[sz >= 5].sum() / sz.sum()), 3), "| keys n>=5:", int((sz >= 5).sum()))
print("ward x ptype cells:", df.groupby(["ward_code_db", "property_type_name"]).ngroups,
      "| cells n<20:", int((df.groupby(["ward_code_db", "property_type_name"]).size() < 20).sum()))
for c in ["floor_count", "bedroom_count", "bathroom_count"]:
    print(c, "max", df[c].max(), "p99.9", df[c].quantile(.999))
