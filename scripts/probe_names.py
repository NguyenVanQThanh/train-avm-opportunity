"""
File:        probe_names.py
Description: Inspect HCM-related names/codes in crosswalk and HF shard0 to design normalization.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

import pandas as pd

cw = pd.read_csv("data/probe/crosswalk.csv", dtype=str)
print("newProvince values:", cw["newProvince"].nunique())
h = cw[cw["newProvince"].str.contains("Hồ Chí Minh", na=False)]
print("HCM new rows:", len(h), "newProvinceCode:", h["newProvinceCode"].unique(), "provinces legacy:", h["provinceShort"].unique())
print("newWardCode sample:", h["newWardCode"].head(5).tolist(), "len set:", h["newWardCode"].str.len().value_counts().to_dict())
print("districtType:", h["districtType"].value_counts().to_dict())
print("district sample:", h["districtShort"].drop_duplicates().head(30).tolist())
print("ward sample numeric:", h[h["wardShort"].str.fullmatch(r"\d+", na=False)]["wardShort"].head(10).tolist())

hf = pd.read_parquet("data/probe/hf/shard_0000.parquet", columns=["province_name", "district_name", "ward_name"])
print("HF province count:", hf["province_name"].nunique())
for p in ["Hồ Chí Minh", "Bình Dương", "Bà Rịa - Vũng Tàu"]:
    s = hf[hf["province_name"] == p]
    print(p, len(s), "districts:", s["district_name"].drop_duplicates().head(25).tolist())
print([p for p in hf["province_name"].unique() if "Bà" in p or "Vũng" in p or "Bình D" in p])
