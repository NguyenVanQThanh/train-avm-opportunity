"""
File:        lgbm.py
Description: Fit model_point (L2 on log price), model_q05, model_q95 (quantile) on identical features, then CQR.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude

Roles: train minus its last `early_stop_holdout_days` fits the trees; that holdout only picks the
iteration count; val calibrates the conformal offsets; test_a is touched by evaluation only.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from avm_train import config
from avm_train.config import stage_dir
from avm_train.evaluate import metrics
from avm_train.features.schema import CATEGORICAL, FEATURES
from avm_train.train import cqr


def _objectives(alpha: float) -> dict:
    return {
        "point": {"objective": "regression"},
        "q05": {"objective": "quantile", "alpha": alpha / 2},
        "q95": {"objective": "quantile", "alpha": 1 - alpha / 2},
    }


def split_holdout(train: pd.DataFrame, days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last `days` days of train (by published_at) become the early-stopping holdout."""
    t = pd.to_datetime(train["_published_at"])
    cut = t.max() - pd.Timedelta(days=days)
    return train[t <= cut], train[t > cut]


def load_models(model_dir) -> dict[str, lgb.Booster]:
    """Loads model_{point,q05,q95}.txt saved by run (files hold only the best iterations)."""
    return {n: lgb.Booster(model_file=str(model_dir / f"model_{n}.txt")) for n in ("point", "q05", "q95")}


def fit_models(fit_df: pd.DataFrame, stop_df: pd.DataFrame, cfg: dict,
               names: tuple[str, ...] | None = None) -> dict[str, lgb.Booster]:
    """Trains the boosters (all three, or only `names`) with identical features, params and seed."""
    mc = cfg["model"]
    models = {}
    for name, obj in _objectives(cfg["cqr"]["alpha"]).items():
        if names is not None and name not in names:
            continue
        params = {**mc["params"], **obj, "seed": cfg["seed"]}
        dtrain = lgb.Dataset(fit_df[FEATURES], fit_df["y"], categorical_feature=CATEGORICAL, free_raw_data=False)
        dstop = lgb.Dataset(stop_df[FEATURES], stop_df["y"], categorical_feature=CATEGORICAL, reference=dtrain)
        models[name] = lgb.train(
            params, dtrain, mc["num_boost_round"], valid_sets=[dstop],
            callbacks=[lgb.early_stopping(mc["early_stopping_rounds"], verbose=False)],
        )
    return models


def predict_log(models: dict, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (lo, point, hi) in log space with lo <= point <= hi enforced (quantiles can cross)."""
    x = df[FEATURES].to_numpy(dtype="float64")
    point, lo, hi = (_predict_safe(models[n], x) for n in ("point", "q05", "q95"))
    return np.minimum(lo, point), point, np.maximum(hi, point)


# lightgbm 4.7.0 on Windows intermittently hits an access violation in multi-threaded
# Booster.predict (reproduced on 140k rows, also with 20k-row chunks); one thread is stable.
def _predict_safe(booster: lgb.Booster, x: np.ndarray) -> np.ndarray:
    """Booster.predict on the best iteration, single-threaded."""
    return booster.predict(x, num_iteration=booster.best_iteration, num_threads=1)


def run(cfg: dict, args=None) -> None:
    """Runs stage `train`: point + q05 + q95 models, CQR calibration, predictions on val and test_a.

    Process:
        1. Hold out the newest train days for early stopping and fit the three models.
        2. Save the models at their best iteration.
        3. Calibrate CQR offsets on val.
        4. Predict val and test_a with calibrated intervals and compute metrics.
        5. Write train_report.json and print a summary.
    """
    src, out = stage_dir(cfg, "features"), stage_dir(cfg, "train")
    parts = {k: pd.read_parquet(src / f"{k}.parquet") for k in ("train", "val", "test_a")}
    mc, qc = cfg["model"], cfg["cqr"]

    # 1. Newest days stop training so early stopping does not tune on val, which CQR needs untouched.
    fit_df, stop_df = split_holdout(parts["train"], mc["early_stop_holdout_days"])
    if len(stop_df) == 0:
        raise config.GateError("early-stopping holdout is empty; check split.train dates")
    models = fit_models(fit_df, stop_df, cfg)

    # 2. Persist models.
    (out / "models").mkdir(exist_ok=True)
    for name, b in models.items():
        b.save_model(str(out / "models" / f"model_{name}.txt"), num_iteration=b.best_iteration)

    # 3. Val is the calibration set: its residuals set the interval widening.
    val = parts["val"]
    lo, point, hi = predict_log(models, val)
    conf = cqr.fit(lo, hi, val["y"].to_numpy(), val["_ptype"].to_numpy(), val["_ward"].to_numpy(),
                   qc["alpha"], qc["min_segment_rows"])
    conf["config_sha256"] = config.sha256(cfg)
    (out / "conformal.json").write_text(json.dumps(conf, ensure_ascii=False, indent=1), encoding="utf-8")

    # 4. Report raw and calibrated coverage side by side to show what CQR changed.
    report = {
        "config_sha256": config.sha256(cfg),
        "rows": {"fit": len(fit_df), "early_stop": len(stop_df), "calibration": len(val)},
        "best_iteration": {n: b.best_iteration for n, b in models.items()},
        "q_hat_global": conf["global"],
    }
    for part in ("val", "test_a"):
        df = parts[part]
        lo, point, hi = predict_log(models, df)
        clo, chi = cqr.apply(conf, lo, hi, df["_ptype"].to_numpy(), df["_ward"].to_numpy())
        price = df["_price"].to_numpy()
        pred = df.assign(_pred=np.exp(point))
        report[part] = {
            **metrics.sliced(pred),
            "interval_raw": metrics.coverage(price, np.exp(lo), np.exp(hi)),
            "interval_cqr": metrics.coverage(price, np.exp(clo), np.exp(chi)),
        }
        pd.DataFrame({"pred": np.exp(point), "lo": np.exp(clo), "hi": np.exp(chi)}).to_parquet(
            out / f"pred_{part}.parquet", index=False)
    # 5. Persist report and print.
    (out / "train_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for part in ("val", "test_a"):
        g = report[part]["global"]
        print(f"model {part}: MdAPE={g['mdape']:.3f} PPE15={g['ppe15']:.3f} COD={g['cod']:.1f} PRD={g['prd']:.3f} "
              f"cov raw={report[part]['interval_raw']['coverage']:.3f} cqr={report[part]['interval_cqr']['coverage']:.3f}")
    print(f"best_iteration={report['best_iteration']}")
