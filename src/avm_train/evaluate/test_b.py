"""
File:        test_b.py
Description: Test B: V-Nexus listings (SALE, ACTIVE) scored by the trained model, compared with the asking price.

Created at:  2026-10-29
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude

Out-of-distribution check only, never used for training. Properties carry no ward_id in the dev DB,
so the ward is read from the last-but-one comma part of full_address:
  1. the name is a current ward name -> wards.code directly;
  2. else it is a pre-2025 ward name -> resolved through the admin crosswalk, and when several old
     wards share the name (e.g. 'Phường 12') the one nearest the listing coordinates wins;
  3. else the row is dropped and counted, never guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from avm_train.extract.crosswalk import map_to_db, name_key
from avm_train.extract.vnexus_db import connect, fetch_wards

_QUERY = """
SELECT l.id, l.price, l.published_at, p.area_m2, p.bedrooms, p.bathrooms, p.floors, p.width_m,
       p.road_width, p.house_direction, p.street_name, p.full_address, p.latitude, p.longitude, pt.code
FROM listings l
JOIN properties p ON p.id = l.property_id
LEFT JOIN property_types pt ON pt.id = p.property_type_id
WHERE l.deleted_at IS NULL AND l.listing_type = 'SALE' AND l.status = 'ACTIVE'
"""

# property_types.code -> training ptype_group; codes absent here are out of scope for the model.
PTYPE_BY_CODE = {
    "can-ho-chung-cu": "Căn hộ chung cư",
    "penthouse": "Căn hộ chung cư",
    "nha-o": "Nhà",
    "nha-biet-thu-doc-lap": "Biệt thự/Nhà liền kề",
    "nha-biet-thu-lien-ke": "Biệt thự/Nhà liền kề",
    "shophouse": "Shophouse",
    "dat-tho-cu": "Đất",
    "dat-nen-du-an": "Đất",
}

_PAREN = re.compile(r"\s*\(.*?\)\s*")


# An old ward further than this from every same-name candidate is treated as unresolved.
MAX_OLD_WARD_KM = 6.0
_HINT_PREFIX = re.compile(r"^(?:thành phố|thị xã|quận|huyện)\s+")


def _km(lat, lon, lats: pd.Series, lons: pd.Series) -> pd.Series:
    """Equirectangular distance in km; accurate to well under 1% at city scale."""
    dy = (lats - lat) * 110.57
    dx = (lons - lon) * 111.32 * np.cos(np.radians(lat))
    return np.sqrt(dx ** 2 + dy ** 2)


@dataclass(frozen=True)
class WardIndex:
    """Lookup from address ward text (+ coordinates) to V-Nexus wards.code."""

    current: dict       # name_key of a current ward -> wards.code
    old: dict           # name_key of a pre-2025 ward -> crosswalk rows (with code_db + centroids)
    centroid: pd.DataFrame  # wards.code -> (lat, lon) of the new ward

    @staticmethod
    def build(cw: pd.DataFrame, wards: pd.DataFrame, target: str, db_province: str) -> WardIndex:
        """cw: raw crosswalk CSV; wards: fetch_wards() frame."""
        mapping, _ = map_to_db(cw, wards, target, db_province)
        to_db = dict(zip(mapping["ward_code_new"], mapping["ward_code_db"]))
        d = cw[cw["newProvinceCode"] == target].copy()
        d["code_db"] = d["newWardCode"].str.zfill(5).map(to_db)
        d = d.dropna(subset=["code_db"])
        for c in ("wardLat", "wardLon", "newWardLat", "newWardLon"):
            d[c] = pd.to_numeric(d[c])
        d["wkey"], d["dkey"] = d["ward"].map(name_key), d["district"].map(name_key)
        w = wards[wards["province_code"].astype(str) == db_province]
        centroid = d.drop_duplicates("code_db").set_index("code_db")[["newWardLat", "newWardLon"]]
        return WardIndex(dict(zip(w["name"].map(name_key), w["code"])),
                         {k: g for k, g in d.groupby("wkey")}, centroid)

    def resolve(self, address, lat, lon) -> tuple[str | None, str]:
        """Returns (wards.code or None, method) where method names the rule that decided.

        Process:
            1. Take the ward name from the second-to-last address part.
            2. Match a current ward name directly.
            3. Otherwise look the name up among old wards and narrow by the district hint.
            4. Accept a unique target ward; else pick the nearest old ward by coordinates.
        """
        # 1. The DB ward columns are empty, so the address text is the only source.
        if not isinstance(address, str):
            return None, "no_address"
        parts = [s.strip() for s in address.split(",")]
        if len(parts) < 2:
            return None, "no_address"
        raw = parts[-2]
        key = name_key(_PAREN.sub("", raw))
        # 2. Current names need no crosswalk.
        if key in self.current:
            return self.current[key], "current_name"
        # 3. Old names are ambiguous across districts ("Phường 12" exists many times).
        g = self.old.get(key)
        if g is None:
            return None, "unknown_name"
        # '(Quận Thủ Đức cũ)' narrows same-name old wards to that old district.
        hint = _PAREN.search(raw)
        if hint:
            h = _HINT_PREFIX.sub("", name_key(hint.group().strip(" ()")).removesuffix(" cũ").strip())
            sub = g[g["dkey"].str.replace(_HINT_PREFIX, "", regex=True) == h]
            g = sub if len(sub) else g
        # 4. Coordinates are the tiebreaker; beyond MAX_OLD_WARD_KM a guess would be worse than a drop.
        if g["code_db"].nunique() == 1:
            return g["code_db"].iloc[0], "old_name_unique"
        if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
            return None, "ambiguous_no_coords"
        olds = g.drop_duplicates("dkey")
        dist = _km(lat, lon, olds["wardLat"], olds["wardLon"])
        if dist.min() > MAX_OLD_WARD_KM:
            return None, "ambiguous_too_far"
        rows = g[g["dkey"] == olds.loc[dist.idxmin(), "dkey"]]
        if rows["code_db"].nunique() > 1:  # old ward split between several new wards
            rows = rows.assign(d=_km(lat, lon, rows["newWardLat"], rows["newWardLon"]))
            rows = rows.sort_values("d")
        return rows["code_db"].iloc[0], "old_name_nearest"


def to_listing_frame(rows: pd.DataFrame, index: WardIndex) -> tuple[pd.DataFrame, dict]:
    """Maps DB rows to the training listing columns; returns (frame, drop counts).

    Process:
        1. Resolve each row's ward from its address.
        2. Build the frame in the training column layout.
        3. Count what is lost and why.
        4. Keep only rows with a ptype in scope, a resolved ward and positive price/area.
    """
    # 1. Ward resolution.
    resolved = [index.resolve(a, la, lo) for a, la, lo in zip(rows["full_address"], rows["latitude"], rows["longitude"])]
    # 2. Same column names as the HF listings so the training feature code is reused unchanged.
    df = pd.DataFrame(
        {
            "price": pd.to_numeric(rows["price"]).astype("float64"),
            "area": pd.to_numeric(rows["area_m2"]).astype("float64"),
            "bedroom_count": pd.to_numeric(rows["bedrooms"]).astype("float64"),
            "bathroom_count": pd.to_numeric(rows["bathrooms"]).astype("float64"),
            "floor_count": pd.to_numeric(rows["floors"]).astype("float64"),
            "frontage_width": pd.to_numeric(rows["width_m"]).astype("float64"),
            "road_width": pd.to_numeric(rows["road_width"]).astype("float64"),
            "house_direction": rows["house_direction"],
            "street_name": rows["street_name"],
            # The DB has no project column on properties; the feature is false for every Test B row.
            "project_name": pd.Series([None] * len(rows), index=rows.index, dtype="object"),
            "property_type_name": rows["code"].map(PTYPE_BY_CODE),
            "ward_code_db": [r[0] for r in resolved],
            "ward_confidence": "exact",
            "published_at": pd.to_datetime(rows["published_at"], utc=True),
        },
        index=rows.index,
    )
    # 3. Drop reasons are reported so a thin Test B is not mistaken for a clean one.
    drops = {
        "rows_in": len(df),
        "no_ptype_in_scope": int(df["property_type_name"].isna().sum()),
        "ward_unmatched": int(df["ward_code_db"].isna().sum()),
        "ward_method": pd.Series([r[1] for r in resolved]).value_counts().to_dict(),
        "listing_to_ward_km": _geo_check(rows, df, index),
        "no_area_or_price": int((df["area"].isna() | df["price"].isna()).sum()),
    }
    # 4. Filter.
    keep = df["property_type_name"].notna() & df["ward_code_db"].notna() & (df["area"] > 0) & (df["price"] > 0)
    drops["rows_kept"] = int(keep.sum())
    return df[keep].reset_index(drop=True), drops


def _geo_check(rows: pd.DataFrame, df: pd.DataFrame, index: WardIndex) -> dict | None:
    """Sanity check of the address match: km from listing coordinates to its resolved ward centroid."""
    ok = (df["ward_code_db"].notna() & rows["latitude"].notna() & rows["longitude"].notna()
          & df["ward_code_db"].isin(index.centroid.index)).to_numpy()
    if not ok.any():
        return None
    c = index.centroid.loc[df.loc[ok, "ward_code_db"]]
    d = _km(rows.loc[ok, "latitude"].astype(float).to_numpy(), rows.loc[ok, "longitude"].astype(float).to_numpy(),
            c["newWardLat"].to_numpy(), c["newWardLon"].to_numpy())
    return {"n": int(ok.sum()), "median": float(np.round(np.median(d), 2)),
            "p90": float(np.round(np.quantile(d, 0.9), 2)), "share_over_10km": float(np.round((d > 10).mean(), 4))}


def load(cw: pd.DataFrame, target: str, province_code_db: str) -> tuple[pd.DataFrame, dict]:
    """Reads Test B listings from the DB (read-only) and maps them to training columns.

    cw is the raw admin crosswalk (data/<run>/raw/crosswalk.csv) used to resolve pre-2025 ward names.
    """
    index = WardIndex.build(cw, fetch_wards(), target, province_code_db)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(_QUERY)
        cols = ["id", "price", "published_at", "area_m2", "bedrooms", "bathrooms", "floors", "width_m",
                "road_width", "house_direction", "street_name", "full_address", "latitude", "longitude", "code"]
        rows = pd.DataFrame(cur.fetchall(), columns=cols)
    return to_listing_frame(rows, index)


def unmatched_share(drops: dict) -> float:
    """Share of input rows lost; reported so a low-coverage Test B is not read as a clean one."""
    return float(np.round(1 - drops["rows_kept"] / max(drops["rows_in"], 1), 4))
