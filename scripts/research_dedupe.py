"""
File:        research_dedupe.py
Description: Is the (ward, street, area, price, day) dedupe removing real repeats or distinct listings?

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

import pandas as pd

df = pd.read_parquet("data/v0-hf/crosswalk/listings.parquet")
df = df[df["price"].between(1e8, 5e11) & df["area"].between(5, 5e4)]
df["day"] = df["published_at"].dt.normalize()
k = ["ward_code_db", "street_name", "area", "price", "day"]
df["street_name"] = df["street_name"].fillna("")
d = df.duplicated(k, keep=False)
print("rows in duplicate groups:", int(d.sum()), "| removed if keep newest:", int(df.duplicated(k).sum()))
print("street_name empty share among removed:", round(float((df[df.duplicated(k)]["street_name"] == "").mean()), 3),
      "| overall:", round(float((df["street_name"] == "").mean()), 3))
sz = df[d].groupby(k).size()
print("group size quantiles:", sz.quantile([.5, .9, .99, 1]).tolist())
# Do duplicates agree on other attributes (supports "same listing")?
cols = ["property_type_name", "bedroom_count", "floor_count", "frontage_width", "house_direction", "project_name"]
agree = df[d].groupby(k)[cols].nunique(dropna=False).le(1).mean().round(3).to_dict()
print("share of dup groups where attribute is identical:", agree)
# Round numbers: area & price both round are likely templated, not unique.
rnd = (df["price"] % 1e8 == 0) & (df["area"] % 5 == 0)
print("round price&area share among removed:", round(float(rnd[df.duplicated(k)].mean()), 3), "| overall", round(float(rnd.mean()), 3))
# Same listing re-posted over several days (cross-day) not caught by day key:
k2 = ["ward_code_db", "street_name", "area", "price"]
print("extra removable if day ignored:", int(df.duplicated(k2).sum() - df.duplicated(k).sum()))
