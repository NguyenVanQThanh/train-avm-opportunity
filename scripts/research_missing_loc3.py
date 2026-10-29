"""
File:        research_missing_loc3.py
Description: Research part 3: can ward be mined from name/description? Validate precision on rows with known ward
             (shard 0).

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

import re

import pandas as pd

from avm_train.extract.crosswalk import norm

df = pd.read_parquet("data/v0-hf/raw/hf/shard_0000.parquet", columns=["name", "description", "province_name", "district_name", "ward_name"])
df = df[df["province_name"].isin(["Hồ Chí Minh", "Bình Dương", "Bà Rịa - Vũng Tàu"])].copy()
text = (df["name"].fillna("") + " . " + df["description"].fillna("")).map(lambda t: norm(t) or "")
# Explicit numeric/word ward mentions: "phuong 14", "p14", "p.14", "phuong tan dinh".
pat = re.compile(r"\b(?:phuong|p)\s*\.?\s*(\d{1,2})\b")
pat_name = re.compile(r"\bphuong ([a-z]+(?: [a-z]+){0,2})")
known = df["ward_name"].notna()
mentions = text.map(lambda t: set(pat.findall(t)))
gold = df["ward_name"].map(norm)
has = mentions.map(bool)
single = mentions.map(len).eq(1)
print("rows", len(df), "| with known ward", known.sum(), "| missing ward", (~known).sum())
for lbl, m in (("known", known), ("missing", ~known)):
    print(f"{lbl}: any numeric ward mention {has[m].mean():.3f} | exactly one {single[m].mean():.3f}")
k = known & single & gold.str.fullmatch(r"\d+", na=False)
pred = mentions[k].map(lambda s: next(iter(s)))
print("precision on known numeric wards with exactly-one mention:", round(float((pred == gold[k]).mean()), 3), "n", int(k.sum()))
kn = known & ~gold.str.fullmatch(r"\d+", na=False)
nm = text[kn].map(lambda t: pat_name.findall(t))
rows = [(g, c) for g, c in zip(gold[kn], nm) if c]
hit = [any(g in cand for cand in c) for g, c in rows]
print("named wards: share with 'phuong <name>' mention", round(len(rows) / int(kn.sum()), 3),
      "| a mention contains the true ward", round(sum(hit) / len(hit), 3), "n", len(hit))
