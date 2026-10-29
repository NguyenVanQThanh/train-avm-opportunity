"""
File:        research_missing_loc.py
Description: Research: how recoverable are in-scope rows that lack district/ward? Reads data/v0-hf/raw only.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

import re

import pandas as pd

from avm_train.extract.crosswalk import norm

raw = pd.read_parquet("data/v0-hf/raw/listings.parquet")
cw = pd.read_csv("data/v0-hf/raw/crosswalk.csv", dtype=str)
legacy = {norm(p) for p in cw.loc[cw["newProvinceCode"] == "79", "provinceShort"].unique()}
s = raw[raw["province_name"].map(norm).isin(legacy)].copy()
d, w = s["district_name"].notna(), s["ward_name"].notna()
print("in scope", len(s))
print("both present", (d & w).sum(), "| district only", (d & ~w).sum(), "| ward only", (~d & w).sum(), "| neither", (~d & ~w).sum())
print("by province, missing share:\n", s.assign(miss=~(d & w)).groupby("province_name")["miss"].agg(["size", "mean"]).round(3).to_string())
print("by month, missing share:\n", s.assign(miss=~(d & w), m=s["published_at"].dt.to_period("M")).groupby("m")["miss"].mean().round(3).to_string())
print("missing rows with street:", s.loc[~(d & w), "street_name"].notna().mean().round(3),
      "| project:", s.loc[~(d & w), "project_name"].notna().mean().round(3))
print("price null share: missing", s.loc[~(d & w), "price"].isna().mean().round(3), "present", s.loc[d & w, "price"].isna().mean().round(3))

# (A) district-only rows: can a street resolve the ward uniquely among complete rows?
full = s[d & w].copy()
full["ck"] = full["district_name"].map(norm) + "|" + full["street_name"].map(norm).fillna("")
g = full[full["street_name"].notna()].groupby("ck")["ward_name"].agg(lambda x: x.map(norm).value_counts(normalize=True).iloc[0])
sizes = full[full["street_name"].notna()].groupby("ck").size()
print("street->ward purity (top ward share) weighted mean:", round(float((g * sizes).sum() / sizes.sum()), 3),
      "| share of (district,street) keys with purity>=0.9:", round(float((g >= 0.9).mean()), 3))

# (B) text mining on name+description would need those columns; they were dropped at fetch. Count patterns in shard0 sample.
