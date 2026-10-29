"""
File:        test_evaluate.py
Description: Tests for evaluation metrics, Test B ward resolution and gate checks.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

import numpy as np
import pandas as pd
import pytest

from avm_train import config
from avm_train.evaluate import gate, metrics, test_b
from avm_train.features import price_ref


def test_point_metrics_hand_computed():
    a = np.array([100.0, 100.0, 100.0, 100.0])
    p = np.array([100.0, 110.0, 120.0, 150.0])
    m = metrics.point_metrics(a, p)
    assert m["mdape"] == pytest.approx(0.15)
    assert m["mape"] == pytest.approx(0.20)
    assert (m["ppe10"], m["ppe15"], m["ppe20"]) == (0.5, 0.5, 0.75)
    assert m["bias"] == pytest.approx(0.15)
    assert m["rmsle"] == pytest.approx(np.sqrt(np.mean(np.log(p / a) ** 2)))


def test_prb_detects_regressive_valuations():
    a = np.array([1e9, 2e9, 4e9, 8e9, 16e9] * 4)
    assert abs(metrics.point_metrics(a, a * 1.05)["prb"]) < 1e-6      # flat error: no bias
    assert metrics.point_metrics(a, a ** 0.9 * 1e9 ** 0.1)["prb"] < -0.01  # high values under-valued


def test_interval_metrics_tails_and_score():
    a = np.array([100.0, 100.0, 100.0, 100.0])
    lo = np.array([90.0, 90.0, 101.0, 90.0])
    hi = np.array([110.0, 110.0, 110.0, 99.0])
    m = metrics.interval_metrics(a, lo, hi, alpha=0.1)
    assert (m["coverage"], m["miss_below"], m["miss_above"]) == (0.5, 0.25, 0.25)


def test_breakdown_skips_small_slices_and_ward_distribution():
    df = pd.DataFrame({"_price": [1e9] * 6, "_pred": [1.1e9] * 6, "_ward": ["a"] * 4 + ["b"] * 2})
    out = metrics.breakdown(df, {"w": df["_ward"]}, None, min_rows=3)
    assert list(out["w"]) == ["a"]
    assert metrics.ward_distribution(df, min_rows=3)["n_wards"] == 1


def test_price_ref_json_roundtrip_gives_same_encoding():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "ward_code_db": rng.choice(["w1", "w2"], n), "property_type_name": rng.choice(["Nhà", "Đất"], n),
        "street_name": rng.choice(["Le Loi", "Hai Ba Trung", None], n),
        "price": rng.uniform(1e9, 5e9, n), "area": rng.uniform(30, 90, n),
    })
    t = price_ref.fit(price_ref.prepare(df), 20, 20.0)
    t2 = price_ref.from_json(price_ref.to_json(t))
    w1, s1 = price_ref.apply(price_ref.prepare(df), t)
    w2, s2 = price_ref.apply(price_ref.prepare(df), t2)
    np.testing.assert_allclose(w1, w2)
    np.testing.assert_allclose(s1, s2)


def _ward_index():
    def r(prov, dist, ward, lat, lon, new, code):
        return {"newProvinceCode": "79", "province": prov, "district": dist, "ward": ward, "wardLat": lat,
                "wardLon": lon, "newWard": new, "newWardCode": code, "newWardLat": lat, "newWardLon": lon}
    cw = pd.DataFrame([
        r("Thành phố Hồ Chí Minh", "Quận Gò Vấp", "Phường 12", "10.83", "106.64", "Phường An Hội Tây", "27001"),
        r("Thành phố Hồ Chí Minh", "Quận Tân Bình", "Phường 12", "10.79", "106.65", "Phường Bảy Hiền", "27002"),
        r("Thành phố Hồ Chí Minh", "Quận Thủ Đức", "Phường Tân Phú", "10.85", "106.80", "Phường Tăng Nhơn Phú", "27003"),
        r("Thành phố Hồ Chí Minh", "Quận 9", "Phường Tân Phú", "10.86", "106.81", "Phường Tăng Nhơn Phú", "27003"),
    ])
    wards = pd.DataFrame({"code": ["W1", "W2", "W3"], "province_code": "29",
                          "name": ["Phường An Hội Tây", "Phường Bảy Hiền", "Phường Tăng Nhơn Phú"]})
    return test_b.WardIndex.build(cw, wards, "79", "29")


def test_test_b_ward_resolution():
    ix = _ward_index()
    city = "Thành phố Hồ Chí Minh"
    assert ix.resolve(f"X, Phường Bảy Hiền, {city}", None, None) == ("W2", "current_name")
    # Same old name in two districts: coordinates pick the nearest old ward.
    assert ix.resolve(f"X, Phường 12, {city}", 10.834, 106.641, ) == ("W1", "old_name_nearest")
    assert ix.resolve(f"X, Phường 12, {city}", 10.792, 106.652) == ("W2", "old_name_nearest")
    assert ix.resolve(f"X, Phường 12, {city}", None, None) == (None, "ambiguous_no_coords")
    assert ix.resolve(f"X, Phường 12, {city}", 11.5, 107.5) == (None, "ambiguous_too_far")
    # The '(Quận ... cũ)' hint resolves without coordinates; '2' must not match 'Quận 12'.
    assert ix.resolve(f"X, Phường 12 (Quận Gò Vấp cũ), {city}", None, None)[0] == "W1"
    assert ix.resolve(f"X, Phường Tân Phú (Quận 9 cũ), {city}", None, None) == ("W3", "old_name_unique")
    assert ix.resolve(f"X, Phường Lạ, {city}", 10.8, 106.7) == (None, "unknown_name")
    assert ix.resolve(None, 10.8, 106.7) == (None, "no_address")
    assert "kho-nha-xuong" not in test_b.PTYPE_BY_CODE


def _cfg():
    return config.load("configs/v0_hf.yaml")


def _report(mdape, ppe15, cov=0.9):
    g = {"mdape": mdape, "ppe15": ppe15, "coverage": cov, "prd": 1.0, "n": 1}
    return {"test_a": {"global": g, "ptype_group": {}, "ward_confidence": {},
                       "lift_vs_baseline": {"mdape_reduction": 0.3}}}


def test_gate_hard_fail_and_soft_warn():
    checks = {c["check"]: c["status"] for c in gate._checks(_cfg(), _report(0.20, 0.40))}
    assert checks["test_a_mdape"] == "FAIL" and checks["test_a_ppe15"] == "FAIL"
    checks = {c["check"]: c["status"] for c in gate._checks(_cfg(), _report(0.14, 0.52, cov=0.80))}
    assert checks["test_a_mdape"] == "pass" and checks["test_a_coverage"] == "warn"


def test_sha_ignores_eval_and_gate_but_not_model_settings():
    cfg = _cfg()
    base = config.sha256(cfg)
    cfg["gate"]["test_a_mdape_max"] = 0.5
    cfg["evaluate"]["min_slice_rows"] = 1
    assert config.sha256(cfg) == base
    cfg["model"]["params"]["num_leaves"] = 7
    assert config.sha256(cfg) != base
