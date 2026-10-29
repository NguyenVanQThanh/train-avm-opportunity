"""
File:        export_wards.py
Description: Export V-Nexus wards (code, name, province_name) to CSV for `avm-train crosswalk --wards-file`.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude

Credentials come from AVM_DB_* env vars. Usage: python scripts/export_wards.py data/wards.csv
"""

import sys

from avm_train.extract.vnexus_db import fetch_wards

df = fetch_wards()
df.to_csv(sys.argv[1], index=False, encoding="utf-8")
print("rows:", len(df), "| provinces:", df["province_name"].nunique())
