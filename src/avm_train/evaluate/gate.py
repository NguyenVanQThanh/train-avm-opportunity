"""
File:        gate.py
Description: Stage `evaluate`: full metric battery on Test A, unseen-ward check, Test B, then the release gate.

Created at:  2026-10-29
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude

Writes evaluate/metrics.json always; raises GateError (exit 2) when a hard gate fails, so a failing
model never reaches export. Soft gates only add warnings.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from avm_train import config
from avm_train.config import GateError, stage_dir
from avm_train.evaluate import metrics, test_b
from avm_train.features import price_ref
from avm_train.features.schema import FEATURES, build_matrix, make_features, with_meta
from avm_train.train import baseline, cqr, lgbm


def _predict(models: dict, conf: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Adds _pred/_lo/_hi (VND) to a feature frame: point model and CQR-widened interval."""
    lo, point, hi = lgbm.predict_log(models, df)
    clo, chi = cqr.apply(conf, lo, hi, df["_ptype"].to_numpy(), df["_ward"].to_numpy())
    return df.assign(_pred=np.exp(point), _lo=np.exp(clo), _hi=np.exp(chi))


def _slices(df: pd.DataFrame, ec: dict) -> dict[str, pd.Series]:
    """Slice labels. Price slices use the predicted price (known at inference), never the actual."""
    week = pd.to_datetime(df["_published_at"]).dt.strftime("%G-W%V")
    return {
        "ptype_group": df["_ptype"],
        "ward_confidence": df["_ward_confidence"],
        "area_bin": pd.cut(df["area_m2"], ec["area_bins_m2"], right=False),
        "predicted_price_quantile": pd.qcut(df["_pred"], ec["price_quantiles"], duplicates="drop"),
        "published_week": week,
    }


def _evaluate_part(df: pd.DataFrame, cfg: dict) -> dict:
    ec = cfg["evaluate"]
    base = df.assign(_pred=baseline.predict(df))
    out = metrics.breakdown(df, _slices(df, ec), cfg["cqr"]["alpha"], ec["min_slice_rows"])
    out["wards"] = metrics.ward_distribution(df, ec["min_ward_rows"])
    out["baseline"] = metrics.point_metrics(base["_price"].to_numpy(), base["_pred"].to_numpy())
    g, b = out["global"], out["baseline"]
    out["lift_vs_baseline"] = {
        "mdape_reduction": 1 - g["mdape"] / b["mdape"],
        "ppe15_gain": g["ppe15"] - b["ppe15"],
    }
    return out


def _unseen_ward(cfg: dict) -> dict:
    """Ward-grouped k-fold: the point model is refit on the other folds' wards, then scores wards it
    never saw. price_ref tables are refit per fold too, so held-out wards fall back to ptype-level
    references exactly as a brand-new ward would at inference.

    Process:
        1. Load train rows and cap boost rounds for the per-fold refits.
        2. For each ward fold: rebuild features without it, refit the point model, score the fold.
        3. Pool the held-out predictions and compute metrics against the baseline.
    """
    ec = cfg["evaluate"]["unseen_ward"]
    # 1. Fewer rounds than the main model: this is a relative check, run once per fold.
    raw = pd.read_parquet(stage_dir(cfg, "split") / "train.parquet").reset_index(drop=True)
    cfg_fold = {**cfg, "model": {**cfg["model"], "num_boost_round": ec["num_boost_round"]}}
    parts = []
    # 2. Encodings and codes are rebuilt per fold so the held-out wards leave no trace in them.
    for k in sorted(raw["fold"].unique()):
        tr, held = raw[raw["fold"] != k], raw[raw["fold"] == k]
        mat, others, _, _ = make_features(tr, {"held": held}, cfg_fold)
        fit_df, stop_df = lgbm.split_holdout(mat, cfg["model"]["early_stop_holdout_days"])
        point = lgbm.fit_models(fit_df, stop_df, cfg_fold, names=("point",))["point"]
        h = others["held"]
        h = h.assign(_pred=np.exp(lgbm._predict_safe(point, h[FEATURES].to_numpy(dtype="float64"))))
        parts.append(h)
        print(f"unseen-ward fold {k}: n={len(h):,} iters={point.best_iteration}", flush=True)
    # 3. Pooled metrics.
    df = pd.concat(parts, ignore_index=True)
    out = metrics.breakdown(df, {"ptype_group": df["_ptype"]}, None, cfg["evaluate"]["min_slice_rows"])
    out["baseline"] = metrics.point_metrics(df["_price"].to_numpy(), baseline.predict(df))
    out["num_boost_round"] = ec["num_boost_round"]
    return out


def _test_b(cfg: dict, models: dict, conf: dict, schema: dict) -> dict:
    """Scores V-Nexus DB listings with the trained models (Test B); skipped when not configured.

    Process:
        1. Skip when disabled or when no DB connection is configured.
        2. Load DB listings, resolve wards and keep rows in the training scope.
        3. Skip when too few rows remain for meaningful metrics.
        4. Encode with the shipped tables, predict and compute metrics.
    """
    tb = cfg["evaluate"]["test_b"]
    # 1. Skips are reported in metrics.json, never silent.
    if not tb["enabled"]:
        return {"status": "skipped", "reason": "disabled in config"}
    if "AVM_DB_HOST" not in os.environ:
        return {"status": "skipped", "reason": "AVM_DB_* not set"}
    # 2. Same price/area scope as training, otherwise the comparison is unfair.
    sc = cfg["scope"]
    cw = pd.read_csv(stage_dir(cfg, "raw") / "crosswalk.csv", dtype=str)
    rows, drops = test_b.load(cw, sc["province_code_new"], str(sc["province_code_db"]))
    rows = rows[rows["price"].between(*sc["price_vnd"]) & rows["area"].between(*sc["area_m2"])]
    drops["rows_in_scope"] = len(rows)
    # 3. Too small a sample would give noisy metrics that look authoritative.
    if len(rows) < tb["min_rows"]:
        return {"status": "skipped", "reason": f"only {len(rows)} usable rows", "drops": drops}
    # 4. Inference path: shipped tables and codes, not refit ones.
    tables = price_ref.from_json(schema["price_ref"])
    w, s = price_ref.apply(price_ref.prepare(rows), tables)
    df = with_meta(build_matrix(rows, w, s, schema["category_codes"]), rows)
    df = _predict(models, conf, df)
    out = _evaluate_part(df, cfg)
    return {"status": "ok", "drops": drops, "share_rows_lost": test_b.unmatched_share(drops), **out}


def _check(name: str, level: str, ok: bool, value, limit) -> dict:
    return {"check": name, "level": level, "status": "pass" if ok else ("FAIL" if level == "hard" else "warn"),
            "value": value, "limit": limit}


def _checks(cfg: dict, report: dict) -> list[dict]:
    gc, ta = cfg["gate"], report["test_a"]
    g = ta["global"]
    out = [
        _check("test_a_mdape", "hard", g["mdape"] <= gc["test_a_mdape_max"], g["mdape"], gc["test_a_mdape_max"]),
        _check("test_a_ppe15", "hard", g["ppe15"] >= gc["test_a_ppe15_min"], g["ppe15"], gc["test_a_ppe15_min"]),
        _check("test_a_beats_baseline", "soft", ta["lift_vs_baseline"]["mdape_reduction"] > 0,
               ta["lift_vs_baseline"]["mdape_reduction"], 0),
    ]
    lo, hi = gc["coverage_range"]
    out.append(_check("test_a_coverage", "soft", lo <= g["coverage"] <= hi, g["coverage"], gc["coverage_range"]))
    plo, phi = gc["prd_range"]
    out.append(_check("test_a_prd_global", "soft", plo <= g["prd"] <= phi, g["prd"], gc["prd_range"]))
    for name, m in ta["ptype_group"].items():
        if m["n"] >= gc["segment_min_rows"]:
            out.append(_check(f"test_a_prd[{name}]", "soft", plo <= m["prd"] <= phi, m["prd"], gc["prd_range"]))
    conf = ta["ward_confidence"]
    if {"exact", "default_split"} <= set(conf):
        gap = conf["default_split"]["mdape"] - conf["exact"]["mdape"]
        out.append(_check("default_split_gap", "soft", gap <= gc["default_split_gap_warn"], gap,
                          gc["default_split_gap_warn"]))
    uw = report.get("unseen_ward")
    if uw:
        out.append(_check("unseen_ward_mdape", "soft", uw["global"]["mdape"] <= gc["unseen_ward_mdape_warn"],
                          uw["global"]["mdape"], gc["unseen_ward_mdape_warn"]))
    tb = report.get("test_b", {})
    if tb.get("status") == "ok":
        out.append(_check("test_b_mdape", "soft", tb["global"]["mdape"] <= gc["test_b_mdape_warn"],
                          tb["global"]["mdape"], gc["test_b_mdape_warn"]))
    return out


def run(cfg: dict, args=None) -> None:
    """Runs stage `evaluate`: metric battery, unseen-ward check, Test B, then the gate.

    Process:
        1. Load models, conformal offsets and feature schema; refuse a config-hash mismatch.
        2. Score val and test_a and compute the metric battery.
        3. Run the unseen-ward check and Test B.
        4. Evaluate hard and soft gate checks and write metrics.json.
        5. Raise GateError if a hard check failed.
    """
    feat, train_dir, out = stage_dir(cfg, "features"), stage_dir(cfg, "train"), stage_dir(cfg, "evaluate")
    # 1. Artifacts from another config would be evaluated against the wrong thresholds.
    models = lgbm.load_models(train_dir / "models")
    conf = json.loads((train_dir / "conformal.json").read_text(encoding="utf-8"))
    schema = json.loads((feat / "feature_schema.json").read_text(encoding="utf-8"))
    sha = config.sha256(cfg)
    for name, found in (("conformal.json", conf.get("config_sha256")),
                        ("feature_schema.json", schema["encoding"]["config_sha256"])):
        if found != sha:
            raise GateError(f"{name} was built with a different config; rerun earlier stages")

    # 2. Val and test_a go through the same scoring path.
    report: dict = {"config_sha256": sha, "run_id": cfg["run_id"]}
    for part in ("val", "test_a"):
        df = _predict(models, conf, pd.read_parquet(feat / f"{part}.parquet"))
        report[part] = _evaluate_part(df, cfg)
        print(f"{part}: n={len(df):,} done", flush=True)
    # 3. Both are soft evidence; they never block on their own.
    report["unseen_ward"] = _unseen_ward(cfg) if cfg["evaluate"]["unseen_ward"]["enabled"] else None
    report["test_b"] = _test_b(cfg, models, conf, schema)
    # 4. Write the report before raising so a failing run keeps its evidence.
    report["checks"] = _checks(cfg, report)
    failed = [c for c in report["checks"] if c["status"] == "FAIL"]
    report["verdict"] = "FAIL" if failed else "PASS"

    (out / "metrics.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    _print_summary(report)
    # 5. Only hard checks fail the stage.
    if failed:
        raise GateError("hard gate failed: " + ", ".join(c["check"] for c in failed))


def _print_summary(r: dict) -> None:
    def line(tag, m):
        print(f"{tag:<22} n={m['n']:>7,} MdAPE={m['mdape']:.3f} PPE10={m['ppe10']:.3f} PPE15={m['ppe15']:.3f} "
              f"PPE20={m['ppe20']:.3f} COD={m['cod']:.1f} PRD={m['prd']:.3f} PRB={m['prb'] or 0:+.3f} "
              f"bias={m['bias']:+.3f} RMSLE={m['rmsle']:.3f}")
    for part in ("val", "test_a"):
        line(f"{part} model", r[part]["global"])
        line(f"{part} baseline", r[part]["baseline"])
        iv = r[part]["global"]
        print(f"{part} interval: cov={iv['coverage']:.3f} below={iv['miss_below']:.3f} above={iv['miss_above']:.3f} "
              f"width={iv['median_rel_width']:.2f} score={iv['interval_score_log']:.3f}")
    if r["unseen_ward"]:
        line("unseen-ward model", r["unseen_ward"]["global"])
        line("unseen-ward baseline", r["unseen_ward"]["baseline"])
    tb = r["test_b"]
    print("test_b:", tb["status"], tb.get("reason", ""))
    if tb["status"] == "ok":
        line("test_b model", tb["global"])
    for c in r["checks"]:
        print(f"[{c['status']:<4}] {c['check']}: {c['value']} (limit {c['limit']})")
    print("verdict:", r["verdict"])
