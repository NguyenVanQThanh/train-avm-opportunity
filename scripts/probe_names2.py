"""
File:        probe_names2.py
Description: Second probe of HCM ward and district naming differences between sources. Read-only.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

import pandas as pd

cw = pd.read_csv("data/probe/crosswalk.csv", dtype=str)
h = cw[cw["newProvinceCode"] == "79"]
q3 = h[h["districtShort"] == "Quận 3"][["ward", "wardShort", "wardType", "newWard", "newWardCode", "isDividedWard", "isDefaultNewWard"]]
print(q3.head(6).to_string())
print("dup (prov,dist,ward) keys:", h.duplicated(["provinceShort", "districtShort", "wardShort"]).sum())
print("divided groups:", h.groupby(["provinceShort", "districtShort", "wardShort"]).size().gt(1).sum())
print("wardType:", h["wardType"].value_counts().to_dict())
hf = pd.read_parquet("data/probe/hf/shard_0000.parquet", columns=["province_name", "district_name", "ward_name"])
s = hf[hf["province_name"] == "Hồ Chí Minh"]
print("HF HCM ward null rate:", s["ward_name"].isna().mean().round(3), "district null:", s["district_name"].isna().mean().round(3))
print(s[s["district_name"] == "3"]["ward_name"].value_counts().head(8).to_dict())
print(s[s["district_name"] == "Thủ Đức"]["ward_name"].value_counts().head(8).to_dict())
