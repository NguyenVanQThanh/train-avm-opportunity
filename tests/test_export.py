"""
File:        test_export.py
Description: Tests for the export stage: refusal on a failed gate, sha256 manifest, golden SHAP.

Created at:  2026-10-29
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

import hashlib
import json
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from test_train import _frame

from avm_train import config
from avm_train.export import artifacts
from avm_train.features.schema import FEATURES
from avm_train.train import lgbm


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """A finished tiny run (models, conformal, schema, metrics, test_a) under a temp data root."""
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(artifacts, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    cfg = config.load("configs/v0_hf.yaml")
    cfg["model"].update(num_boost_round=20, early_stopping_rounds=5)
    cfg["model"]["params"].update(num_leaves=8, min_data_in_leaf=5)
    sha = config.sha256(cfg)

    df = _frame(700, 3)
    df["_ptype"], df["_ward"], df["_price"] = "A", "w1", np.exp(df["y"])
    fit_df, stop_df = lgbm.split_holdout(df, 14)
    models = lgbm.fit_models(fit_df, stop_df, cfg)
    (config.stage_dir(cfg, "train") / "models").mkdir()
    for n, b in models.items():
        b.save_model(str(config.stage_dir(cfg, "train") / "models" / f"model_{n}.txt"), num_iteration=b.best_iteration)

    def write(stage, name, obj):
        (config.stage_dir(cfg, stage) / name).write_text(json.dumps(obj), encoding="utf-8")

    write("train", "conformal.json", {"global": 0.1, "ptype": {}, "ward_ptype": {}, "config_sha256": sha})
    write("features", "feature_schema.json", {"encoding": {"config_sha256": sha}})
    write("evaluate", "metrics.json", {"config_sha256": sha, "verdict": "PASS"})
    write("crosswalk", "crosswalk_report.json", {"ok": True})
    df.to_parquet(config.stage_dir(cfg, "features") / "test_a.parquet", index=False)
    return cfg


def test_manifest_hashes_match_files_and_flags_non_production(run_dir):
    artifacts.run(run_dir)
    out = artifacts.ARTIFACTS_ROOT / run_dir["run_id"]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["NON_PRODUCTION"] is True and manifest["DRAFT"] is False
    assert "manifest.json" not in manifest["files"]
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    assert {"model_point.txt", "model_q05.txt", "model_q95.txt", "golden.csv"} <= set(manifest["files"])


def test_golden_shap_sums_to_log_point_and_replays_from_shipped_model(run_dir):
    artifacts.run(run_dir)
    out = artifacts.ARTIFACTS_ROOT / run_dir["run_id"]
    g = pd.read_csv(out / "golden.csv")
    shap = g[[f"shap_{f}" for f in FEATURES] + ["shap_expected_value"]].sum(axis=1)
    np.testing.assert_allclose(shap, g["log_point"], rtol=1e-6, atol=1e-6)
    booster = lgb.Booster(model_file=str(out / "model_point.txt"))
    np.testing.assert_allclose(booster.predict(g[FEATURES]), g["log_point"], rtol=1e-9)


def test_failed_gate_refused_unless_forced_then_flagged_draft(run_dir):
    path = config.stage_dir(run_dir, "evaluate") / "metrics.json"
    path.write_text(json.dumps({"config_sha256": config.sha256(run_dir), "verdict": "FAIL"}), encoding="utf-8")
    with pytest.raises(config.GateError, match="force-draft"):
        artifacts.run(run_dir)
    artifacts.run(run_dir, SimpleNamespace(force_draft=True))
    manifest = json.loads((artifacts.ARTIFACTS_ROOT / run_dir["run_id"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["DRAFT"] is True


def test_existing_version_is_not_overwritten(run_dir):
    artifacts.run(run_dir)
    with pytest.raises(config.GateError, match="immutable"):
        artifacts.run(run_dir)


def test_stale_stage_output_refused(run_dir):
    path = config.stage_dir(run_dir, "evaluate") / "metrics.json"
    path.write_text(json.dumps({"config_sha256": "old", "verdict": "PASS"}), encoding="utf-8")
    with pytest.raises(config.GateError, match="different config"):
        artifacts.run(run_dir)
