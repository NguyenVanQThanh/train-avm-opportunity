"""
File:        artifacts.py
Description: Write the versioned artifact dir artifacts/<run_id>/ that the BE registers: model_*.txt,
             conformal.json, feature_schema.json, metrics.json, crosswalk_report.json, golden.csv
             (Python predictions + SHAP) and manifest.json (sources, seed, git hash, sha256 per file).

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude

oof_predictions.csv is not produced: it belongs to the production (V-Nexus data) release.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import shutil
import subprocess

import numpy as np
import pandas as pd

from avm_train import config
from avm_train.config import GateError, stage_dir
from avm_train.features.schema import FEATURE_VERSION, FEATURES
from avm_train.train import cqr, lgbm

ARTIFACTS_ROOT = pathlib.Path("artifacts")
GOLDEN_ROWS = 500


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict:
    """Commit of the training code; both fields are None when git or the checkout is unavailable."""
    here = pathlib.Path(__file__).parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True,
                                    text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _golden(cfg: dict, models: dict, conf: dict, test_a: pd.DataFrame) -> pd.DataFrame:
    """Sample of test_a rows with the exact inputs, Python outputs and TreeSHAP of the point model.

    The BE parity check replays these rows through lightgbm4j; SHAP is in log-price space with the
    expected value in the last column.
    """
    sample = test_a.sample(min(GOLDEN_ROWS, len(test_a)), random_state=cfg["seed"]).reset_index(drop=True)
    ptype, ward = sample["_ptype"].to_numpy(), sample["_ward"].to_numpy()
    lo, point, hi = lgbm.predict_log(models, sample)
    clo, chi = cqr.apply(conf, lo, hi, ptype, ward)
    booster = models["point"]
    contrib = booster.predict(sample[FEATURES].to_numpy(dtype="float64"),
                              num_iteration=booster.best_iteration, pred_contrib=True, num_threads=1)
    out = sample[FEATURES].copy()
    out["segment_ptype"], out["segment_ward"] = ptype, ward
    out["price_actual"] = sample["_price"].to_numpy()
    out["log_point"], out["log_lo_raw"], out["log_hi_raw"] = point, lo, hi
    out["q_hat"] = cqr.q_hat_for(conf, ptype, ward)
    out["pred_vnd"], out["lo_vnd"], out["hi_vnd"] = np.exp(point), np.exp(clo), np.exp(chi)
    for i, name in enumerate(FEATURES):
        out[f"shap_{name}"] = contrib[:, i]
    out["shap_expected_value"] = contrib[:, len(FEATURES)]
    return out


def run(cfg: dict, args=None) -> None:
    """Runs stage `export`: assembles artifacts/<run_id>/ from the finished stages.

    Process:
        1. Refuse when evaluate did not pass (unless --force-draft) or any stage used another config.
        2. Refuse to overwrite an existing version.
        3. Copy models and JSON reports into a temporary directory.
        4. Write golden.csv from test_a.
        5. Write manifest.json with provenance and the sha256 of every file, then publish the dir.
    """
    force_draft = bool(getattr(args, "force_draft", False))
    version = cfg["run_id"]
    train_dir, feat = stage_dir(cfg, "train"), stage_dir(cfg, "features")
    evaluate_dir, cw_dir = stage_dir(cfg, "evaluate"), stage_dir(cfg, "crosswalk")

    # 1. A model whose gate failed must never reach the BE as if it were releasable.
    metrics = json.loads((evaluate_dir / "metrics.json").read_text(encoding="utf-8"))
    conf = json.loads((train_dir / "conformal.json").read_text(encoding="utf-8"))
    schema = json.loads((feat / "feature_schema.json").read_text(encoding="utf-8"))
    sha = config.sha256(cfg)
    for name, found in (("metrics.json", metrics.get("config_sha256")),
                        ("conformal.json", conf.get("config_sha256")),
                        ("feature_schema.json", schema["encoding"]["config_sha256"])):
        if found != sha:
            raise GateError(f"{name} was built with a different config; rerun earlier stages")
    if metrics["verdict"] != "PASS" and not force_draft:
        raise GateError(f"evaluate verdict is {metrics['verdict']}; use --force-draft to export a DRAFT")

    # 2. Versions are immutable once published; a new model gets a new run_id.
    dest = ARTIFACTS_ROOT / version
    if dest.exists():
        raise GateError(f"{dest} already exists; artifact versions are immutable, use a new run_id")
    tmp = ARTIFACTS_ROOT / f".{version}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    # 3. Model files keep their LightGBM text format so lightgbm4j loads them as they are.
    for name in ("point", "q05", "q95"):
        shutil.copy(train_dir / "models" / f"model_{name}.txt", tmp / f"model_{name}.txt")
    shutil.copy(train_dir / "conformal.json", tmp / "conformal.json")
    shutil.copy(feat / "feature_schema.json", tmp / "feature_schema.json")
    shutil.copy(evaluate_dir / "metrics.json", tmp / "metrics.json")
    shutil.copy(cw_dir / "crosswalk_report.json", tmp / "crosswalk_report.json")

    # 4. Golden rows go through the shipped model files, not the in-memory boosters.
    models = lgbm.load_models(tmp)
    test_a = pd.read_parquet(feat / "test_a.parquet")
    _golden(cfg, models, conf, test_a).to_csv(tmp / "golden.csv", index=False)

    # 5. The manifest is the only file not hashed: it holds the hashes.
    src = cfg["sources"]
    manifest = {
        "version": version,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "NON_PRODUCTION": bool(cfg.get("non_production", False)),
        "DRAFT": metrics["verdict"] != "PASS",
        "verdict": metrics["verdict"],
        "feature_version": FEATURE_VERSION,
        "seed": cfg["seed"],
        "config_sha256": sha,
        "data_sources": {
            "hf_dataset": src["hf_dataset"],
            "hf_revision": src["hf_revision"],
            "crosswalk_repo": src["crosswalk_repo"],
            "crosswalk_commit": src["crosswalk_commit"],
        },
        "windows": {k: cfg["split"][k] for k in ("train", "val", "test_a")},
        "code": _git_state(),
        "files": {p.name: _sha256_file(p) for p in sorted(tmp.iterdir())},
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(dest)
    print(f"export: {dest} verdict={manifest['verdict']} NON_PRODUCTION={manifest['NON_PRODUCTION']} "
          f"files={len(manifest['files']) + 1}")
