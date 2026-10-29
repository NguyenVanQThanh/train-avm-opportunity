"""
File:        hf.py
Description: Downloads the HF dataset and the crosswalk CSV at pinned revisions; refuses unpinned ones.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import urllib.request

import pandas as pd
from huggingface_hub import snapshot_download

from avm_train.config import GateError, stage_dir

# Free-text (name, description) and mostly-empty columns (house_depth, balcony_direction) are
# dropped at read time: they are not features and description alone is ~half the dataset size.
KEEP = [
    "property_type_name", "province_name", "district_name", "ward_name", "street_name",
    "project_name", "price", "area", "floor_count", "frontage_width", "road_width",
    "bedroom_count", "bathroom_count", "house_direction", "published_at",
]


def run(cfg: dict, args=None) -> None:
    """Runs stage `fetch`: pulls the pinned HF shards and the pinned crosswalk CSV.

    Process:
        1. Refuse to run unless both sources are pinned to a revision.
        2. Download the dataset shards.
        3. Keep only the columns and the date window the pipeline uses.
        4. Download the crosswalk and write fetch_meta.json with hashes and row counts.
    """
    s = cfg["sources"]
    # 1. Unpinned sources make a run irreproducible.
    for key in ("hf_revision", "crosswalk_commit"):
        if not s.get(key):
            raise GateError(f"sources.{key} is not pinned")

    # 2. Download the dataset shards.
    out = stage_dir(cfg, "raw")
    snap = snapshot_download(
        s["hf_dataset"], repo_type="dataset", revision=s["hf_revision"],
        allow_patterns=["shard_*.parquet"], local_dir=out / "hf",
    )

    # 3. Keep the training window only.
    lo = pd.Timestamp(cfg["split"]["train"][0])
    hi = pd.Timestamp(cfg["split"]["test_a"][1]) + pd.Timedelta(days=1)
    frames, per_shard = [], {}
    for f in sorted(pathlib.Path(snap).glob("shard_*.parquet")):
        df = pd.read_parquet(f, columns=KEEP)
        n = len(df)
        # price arrives as text in the source; unparsable values become NaN and are dropped in clean.
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df = df[(df["published_at"] >= lo) & (df["published_at"] < hi)]
        per_shard[f.name] = {"rows": n, "rows_in_window": len(df)}
        frames.append(df)
    listings = pd.concat(frames, ignore_index=True)
    listings.to_parquet(out / "listings.parquet", index=False)

    # 4. Download the crosswalk and record provenance.
    url = (
        f"https://raw.githubusercontent.com/{s['crosswalk_repo']}/"
        f"{s['crosswalk_commit']}/{s['crosswalk_file']}"
    )
    cw = out / "crosswalk.csv"
    urllib.request.urlretrieve(url, cw)

    meta = {
        "fetched_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "hf_dataset": s["hf_dataset"],
        "hf_revision": s["hf_revision"],
        "crosswalk_url": url,
        "crosswalk_sha256": hashlib.sha256(cw.read_bytes()).hexdigest(),
        "window": [str(lo.date()), str((hi - pd.Timedelta(days=1)).date())],
        "rows_total": int(sum(v["rows"] for v in per_shard.values())),
        "rows_in_window": len(listings),
        "shards": per_shard,
    }
    (out / "fetch_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"fetched {meta['rows_in_window']:,} of {meta['rows_total']:,} rows")
