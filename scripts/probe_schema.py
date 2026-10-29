"""
File:        probe_schema.py
Description: Download crosswalk CSV + first HF shard (cached under data/probe/) and print schemas.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

import pathlib
import urllib.request

import pandas as pd
from huggingface_hub import hf_hub_download

OUT = pathlib.Path("data/probe")
OUT.mkdir(parents=True, exist_ok=True)

URL = (
    "https://raw.githubusercontent.com/tranngocminhhieu/vietnamadminunits/"
    "7fac8c45805aad9916b17237c54baf4502303b93/data/processed/"
    "convert_legacy_2025_with_location_and_default_ward.csv"
)
cw = OUT / "crosswalk.csv"
if not cw.exists():
    urllib.request.urlretrieve(URL, cw)
df = pd.read_csv(cw, dtype=str)
print("CROSSWALK rows", len(df))
print(df.dtypes.to_string())
print(df.head(3).T.to_string())
for c in df.columns:
    if "divided" in c.lower() or "default" in c.lower():
        print(c, df[c].value_counts(dropna=False).to_dict())

p = hf_hub_download(
    "tinixai/vietnam-real-estates", "shard_0000.parquet", repo_type="dataset",
    revision="ca3fedcbb089bf65f7e4cfb03316cce4bd0d779f", local_dir=OUT / "hf",
)
h = pd.read_parquet(p)
print("HF shard0 rows", len(h))
print(h.dtypes.to_string())
print(h.head(3).T.to_string())
print(h.isna().mean().round(3).to_string())
