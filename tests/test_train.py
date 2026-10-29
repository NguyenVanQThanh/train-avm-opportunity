"""
File:        test_train.py
Description: Tests for model training, conformal calibration and the config snapshot.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

import numpy as np
import pandas as pd
import pytest

from avm_train import config
from avm_train.evaluate import metrics
from avm_train.features.schema import FEATURES
from avm_train.train import cqr, lgbm


def test_point_metrics_perfect_and_off():
    a = np.array([100.0, 200.0, 300.0, 400.0])
    m = metrics.point_metrics(a, a)
    assert (m["mdape"], m["ppe10"], m["cod"]) == (0.0, 1.0, 0.0)
    m = metrics.point_metrics(a, a * np.array([1.0, 1.0, 1.2, 1.5]))
    assert m["ppe10"] == 0.5 and m["ppe15"] == 0.5
    assert m["mdape"] == pytest.approx(0.1)


def test_cqr_restores_coverage_per_segment():
    rng = np.random.default_rng(0)
    n = 4000
    ptype = np.where(rng.random(n) < 0.5, "A", "B")
    ward = np.full(n, "w1")
    noise = rng.normal(0, np.where(ptype == "A", 0.1, 0.4))  # B much noisier than the quantiles claim
    y = noise
    lo, hi = np.full(n, -0.16), np.full(n, 0.16)  # same narrow interval for both segments
    conf = cqr.fit(lo[:2000], hi[:2000], y[:2000], ptype[:2000], ward[:2000], alpha=0.1, min_rows=100)
    clo, chi = cqr.apply(conf, lo[2000:], hi[2000:], ptype[2000:], ward[2000:])
    for p in ("A", "B"):
        s = ptype[2000:] == p
        cov = ((y[2000:][s] >= clo[s]) & (y[2000:][s] <= chi[s])).mean()
        assert cov == pytest.approx(0.9, abs=0.04)


def test_cqr_sparse_segment_falls_back_to_ptype():
    conf = {"global": 9.0, "ptype": {"A": 1.0}, "ward_ptype": {"w1|A": 0.5}}
    q = cqr.q_hat_for(conf, np.array(["A", "A", "Z"]), np.array(["w1", "w2", "w1"]))
    assert q.tolist() == [0.5, 1.0, 9.0]


def _frame(n, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.random(n) for c in FEATURES})
    for c in ("ptype_group", "house_direction", "ward_code_db"):
        df[c] = rng.integers(0, 3, n).astype("float64")
    df["y"] = 3 * df["area_m2"] + df["ward_price_ref"] + rng.normal(0, 0.1, n)
    df["_published_at"] = pd.Timestamp("2025-06-01") + pd.to_timedelta(rng.integers(0, 60, n), unit="D")
    return df


def test_train_reload_predicts_identically_and_quantiles_ordered(tmp_path):
    import lightgbm as lgb

    cfg = config.load("configs/v0_hf.yaml")
    cfg["model"].update(num_boost_round=30, early_stopping_rounds=5)
    cfg["model"]["params"].update(num_leaves=8, min_data_in_leaf=5)
    df = _frame(600, 1)
    fit_df, stop_df = lgbm.split_holdout(df, 14)
    assert fit_df["_published_at"].max() < stop_df["_published_at"].min()
    models = lgbm.fit_models(fit_df, stop_df, cfg)
    lo, point, hi = lgbm.predict_log(models, df)
    assert (lo <= point).all() and (point <= hi).all()
    models["point"].save_model(str(tmp_path / "m.txt"), num_iteration=models["point"].best_iteration)
    again = lgb.Booster(model_file=str(tmp_path / "m.txt")).predict(df[FEATURES])
    np.testing.assert_allclose(again, point, rtol=1e-9)


def test_snapshot_refuses_changed_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    cfg = config.load("configs/v0_hf.yaml")
    config.snapshot(cfg)
    config.snapshot(cfg)
    cfg["model"]["params"]["learning_rate"] = 0.5
    with pytest.raises(config.GateError, match="new run_id"):
        config.snapshot(cfg)
