"""
File:        research_clean.py
Description: Distribution check on crosswalked rows to set cleaning rules. Read-only.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

import pandas as pd

df = pd.read_parquet("data/v0-hf/crosswalk/listings.parquet")
print("rows", len(df))
print("ptype:", df["property_type_name"].value_counts(dropna=False).to_dict())
for c in ["price", "area", "floor_count", "frontage_width", "road_width", "bedroom_count", "bathroom_count"]:
    q = df[c].quantile([0, .01, .5, .99, .999, 1]).round(2).tolist()
    print(f"{c:15} null={df[c].isna().mean():.3f} zero={(df[c] == 0).mean():.4f} q0/1/50/99/99.9/100={q}")
print("price<1e8:", (df["price"] < 1e8).mean().round(4), "| price>5e11:", (df["price"] > 5e11).mean().round(5))
print("area<5:", (df["area"] < 5).mean().round(4), "| area>5e4:", (df["area"] > 5e4).mean().round(5))
print("house_direction:", df["house_direction"].value_counts(dropna=False).head(12).to_dict())
print("dtypes:", df.dtypes.astype(str).to_dict())
