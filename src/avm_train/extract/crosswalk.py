"""
File:        crosswalk.py
Description: Legacy (province, district, ward) -> ward_code_new (5 digits) via the admin-unit CSV.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude

Adds ward_code_new, province_code_new and ward_confidence ('exact' | 'default_split') to rows in
scope, drops unmatched rows, writes crosswalk_report.json, and stops when the match rate over
rows that carry a full location falls below crosswalk.min_match_rate.
"""

from __future__ import annotations

import json
import re
import unicodedata

import pandas as pd
from unidecode import unidecode

from avm_train.config import GateError, stage_dir

_NUMERIC = re.compile(r"^(?:quan|phuong)\s+(\d+)$")
_PREFIX = re.compile(r"^(?:thanh pho|thi tran|thi xa|quan|huyen|phuong|xa)\s+")
_WARD_CODE_LEN = 5


def norm(value, strip_prefix: bool = False) -> str | None:
    """Unaccents, lowercases and collapses whitespace; 'Quan 3'/'Phuong 14' become '3'/'14'.

    Administrative prefixes are removed only when strip_prefix is set (second matching pass):
    a bare name such as 'Xa La' would otherwise lose its first word.
    """
    if value is None or pd.isna(value):
        return None
    s = re.sub(r"[^a-z0-9 ]+", " ", unidecode(str(value)).lower().replace("đ", "d"))
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None
    m = _NUMERIC.match(s)
    if m:
        return m.group(1)
    return _PREFIX.sub("", s) if strip_prefix else s


# Combining tone marks (grave, acute, tilde, hook above, dot below).
_TONES = "".join(map(chr, (0x300, 0x301, 0x303, 0x309, 0x323)))
# Old-style placement (ho + tone + a) -> new-style (hoa + tone): sources disagree on which they use.
_OLD_TONE = re.compile(rf"([ou])([{_TONES}])([aey])")


def name_key(value) -> str:
    """Accent-preserving key for joining ward names to the DB.

    Accents stay because stripping them merges distinct wards ('Xã Thanh An' vs 'Xã Thạnh An').
    Tone placement is unified (tone mark moved onto the second vowel of oa/oe/uy) because the DB
    writes 'Hoà' while the crosswalk writes 'Hòa' for the same ward.
    """
    s = unicodedata.normalize("NFD", str(value).lower())
    s = _OLD_TONE.sub(lambda m: m[1] + m[3] + m[2], s)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def map_to_db(cw: pd.DataFrame, wards: pd.DataFrame, target: str, db_province: str) -> tuple[pd.DataFrame, dict]:
    """Maps each new ward of the target province to its V-Nexus wards.code by accent-kept name.

    Returns (mapping[ward_code_new, ward_code_db], problems): problems lists new wards with zero or
    several DB matches, and DB wards of the province that no new ward points to.
    """
    new = (
        cw.loc[cw["newProvinceCode"] == target, ["newWard", "newWardCode"]]
        .drop_duplicates()
        .assign(ward_code_new=lambda d: d["newWardCode"].str.zfill(_WARD_CODE_LEN),
                key=lambda d: d["newWard"].map(name_key))
    )
    db = wards.loc[wards["province_code"].astype(str) == db_province, ["code", "name"]].assign(
        key=lambda d: d["name"].map(name_key)
    )
    counts = db.groupby("key").size()
    joined = new.merge(db.rename(columns={"code": "ward_code_db"}), on="key", how="left")
    joined["n"] = joined["key"].map(counts).fillna(0).astype(int)
    problems = {
        "new_wards_without_exactly_one_db_match": joined.loc[joined["n"] != 1, "newWard"].tolist(),
        "db_wards_not_targeted": sorted(set(db["code"]) - set(joined["ward_code_db"].dropna())),
    }
    return joined.loc[joined["n"] == 1, ["ward_code_new", "ward_code_db"]], problems


def build_lookup(cw: pd.DataFrame, strip_prefix: bool) -> pd.DataFrame:
    """One row per legacy key with its target ward; divided wards resolve to the default new ward.

    Keys whose candidates disagree and have no default new ward are dropped (never guessed).
    """
    k = pd.DataFrame(
        {
            "p": cw["provinceShort"].map(lambda v: norm(v, strip_prefix)),
            "d": cw["district"].map(lambda v: norm(v, strip_prefix)),
            "w": cw["ward"].map(lambda v: norm(v, strip_prefix)),
            "ward_code_new": cw["newWardCode"].str.zfill(_WARD_CODE_LEN),
            "province_code_new": cw["newProvinceCode"],
            "divided": cw["isDividedWard"].eq("True"),
            "default": cw["isDefaultNewWard"].eq("True"),
        }
    ).dropna(subset=["p", "d", "w"])

    rows = []
    for _, g in k.groupby(["p", "d", "w"], sort=False):
        if g["ward_code_new"].nunique() == 1:
            pick, conf = g.iloc[0], ("default_split" if g["divided"].any() else "exact")
        elif g["default"].any():
            pick, conf = g[g["default"]].iloc[0], "default_split"
        else:
            continue
        rows.append(
            (pick["p"], pick["d"], pick["w"], pick["ward_code_new"], pick["province_code_new"], conf)
        )
    return pd.DataFrame(
        rows, columns=["p", "d", "w", "ward_code_new", "province_code_new", "ward_confidence"]
    )


def apply(listings: pd.DataFrame, cw: pd.DataFrame) -> pd.DataFrame:
    """Returns listings plus ward_code_new/province_code_new/ward_confidence (NaN if unmatched).

    Pass 1 matches names as written; pass 2 retries leftovers with administrative prefixes removed.
    """
    loc = ["province_name", "district_name", "ward_name"]
    cols = ["ward_code_new", "province_code_new", "ward_confidence"]
    combos = listings[loc].drop_duplicates().reset_index(drop=True)
    for c in cols:
        combos[c] = None
    for strip in (False, True):
        keys = pd.DataFrame(
            {k: combos[n].map(lambda v, s=strip: norm(v, s)) for k, n in zip("pdw", loc)}
        )
        hit = keys.merge(build_lookup(cw, strip), on=["p", "d", "w"], how="left")
        todo = combos["ward_code_new"].isna().to_numpy()
        for c in cols:
            combos.loc[todo, c] = hit.loc[todo, c].to_numpy()
    return listings.merge(combos, on=loc, how="left")


def run(cfg: dict, args=None) -> None:
    """Runs stage `crosswalk`: maps legacy ward names to the new ward and to V-Nexus wards.code.

    Process:
        1. Select listings whose legacy province merges into the target province.
        2. Match legacy names to the new ward and compute the match rate.
        3. Write the report; fail below the minimum match rate.
        4. Map every new ward 1:1 to wards.code from --wards-file; fail on any gap.
        5. Write the matched listings with ward_code_db.
    """
    raw = stage_dir(cfg, "raw")
    out = stage_dir(cfg, "crosswalk")
    target = cfg["scope"]["province_code_new"]
    min_rate = cfg["crosswalk"]["min_match_rate"]

    cw = pd.read_csv(raw / "crosswalk.csv", dtype=str)
    listings = pd.read_parquet(raw / "listings.parquet")

    # 1. Rows whose legacy province maps into the target province (3 legacy provinces merge into HCM).
    legacy = {norm(p) for p in cw.loc[cw["newProvinceCode"] == target, "provinceShort"].unique()}
    in_scope = listings[listings["province_name"].map(norm).isin(legacy)].copy()
    full = in_scope[["province_name", "district_name", "ward_name"]].notna().all(axis=1)

    # 2. Rate is measured on rows with a full location; missing-location rows cannot be matched at all.
    matched_all = apply(in_scope, cw)
    ok = matched_all["ward_code_new"].notna() & matched_all["province_code_new"].eq(target)
    n_full = int(full.sum())
    n_ok = int((ok & full.to_numpy()).sum())
    rate = n_ok / n_full if n_full else 0.0

    # 3. The report is written before the gate so a failing run still leaves diagnostics.
    unmatched = (
        matched_all[full.to_numpy() & ~ok]
        .groupby(["province_name", "district_name", "ward_name"]).size()
        .sort_values(ascending=False).head(30)
    )
    report = {
        "target_province_code_new": target,
        "legacy_provinces_in_scope": sorted(legacy),
        "rows_in_scope": len(in_scope),
        "rows_missing_location": int((~full).sum()),
        "rows_with_full_location": n_full,
        "rows_matched": n_ok,
        "match_rate_full_location": round(rate, 4),
        "match_rate_all_in_scope": round(n_ok / len(in_scope), 4) if len(in_scope) else 0.0,
        "default_split_share": round(
            float(matched_all.loc[ok, "ward_confidence"].eq("default_split").mean()), 4
        ) if n_ok else 0.0,
        "rows_by_province_code_new": matched_all.loc[ok, "province_code_new"].value_counts().to_dict(),
        "top_unmatched_keys": [
            {"province": p, "district": d, "ward": w, "rows": int(n)}
            for (p, d, w), n in unmatched.items()
        ],
        "wards_db_check": "skipped (no --wards-file)",
        "min_match_rate": min_rate,
    }

    (out / "crosswalk_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if rate < min_rate:
        raise GateError(f"crosswalk match rate {rate:.1%} < {min_rate:.0%}; see crosswalk_report.json")

    # 4. Ward key in training must be the V-Nexus wards.code, so every ward has to map 1:1 to the DB.
    wards_file = getattr(args, "wards_file", None)
    if not wards_file:
        raise GateError("--wards-file is required: ward_code_db is the category key")
    mapping, problems = map_to_db(
        cw, pd.read_csv(wards_file, dtype=str), target, str(cfg["scope"]["province_code_db"])
    )
    report["wards_db_check"] = {
        "mapped_wards": len(mapping),
        "new_wards_without_exactly_one_db_match": problems["new_wards_without_exactly_one_db_match"],
        "db_wards_not_targeted": problems["db_wards_not_targeted"],
    }
    (out / "crosswalk_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if problems["new_wards_without_exactly_one_db_match"] or problems["db_wards_not_targeted"]:
        raise GateError("ward names do not map 1:1 to wards.code; see crosswalk_report.json")

    # 5. Persist only rows that matched into the target province.
    matched_all = matched_all.merge(mapping, on="ward_code_new", how="left")
    matched_all[ok].to_parquet(out / "listings.parquet", index=False)
    print(f"matched {n_ok:,}/{n_full:,} ({rate:.1%}); missing location {report['rows_missing_location']:,}")
