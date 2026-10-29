"""
File:        test_features.py
Description: Tests for feature normalizers and out-of-fold price_ref encodings.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

import numpy as np
import pandas as pd

from avm_train.features import normalize, price_ref, schema


def _listings(n=400, seed=0):
    rng = np.random.default_rng(seed)
    ward = rng.choice(["w1", "w2", "w3"], n)
    base = {"w1": 17.5, "w2": 18.5, "w3": 19.5}
    ppm = np.array([base[w] for w in ward]) + rng.normal(0, 0.2, n)
    area = rng.uniform(30, 120, n)
    return pd.DataFrame(
        {
            "ward_code_db": ward,
            "property_type_name": rng.choice(["Nhà", "Đất"], n),
            "street_name": rng.choice(["Nguyễn Huệ", "Lê Lợi", None], n),
            "area": area,
            "price": np.exp(ppm) * area,
        }
    )


def test_direction_variants_collapse_and_garbage_is_none():
    assert {normalize.normalize_direction(v) for v in ["Đông Nam", "Đông - Nam", "đông  nam"]} == {"dong_nam"}
    assert normalize.normalize_direction("Tây - Bắc") == "tay_bac"
    assert normalize.normalize_direction("Không rõ") is None
    assert normalize.normalize_direction(None) is None


def test_unknown_category_encodes_to_nan_not_new_code():
    out = normalize.encode(pd.Series(["Nhà", "Lạ"]), {"Nhà": 3})
    assert out[0] == 3 and np.isnan(out[1])


def test_oof_encoding_ignores_own_label():
    df = _listings()
    base = price_ref.prepare(df)
    folds = np.random.default_rng(1).integers(0, 5, len(base))
    w0, s0 = price_ref.oof(base, folds, min_rows=20, m=20.0)

    i = 7
    # Shift every row sharing row i's (ward, ptype) cell, so other folds' cell medians must move,
    # while row i's own encoding (fit on the other folds) still cannot depend on its own label.
    cell = (base["ward"] == base.loc[i, "ward"]) & (base["ptype"] == base.loc[i, "ptype"])
    own = np.zeros(len(base), bool)
    own[i] = True
    changed = base.copy()
    changed.loc[own, "ppm"] += 5.0
    w1, s1 = price_ref.oof(changed, folds, min_rows=20, m=20.0)
    assert w1[i] == w0[i] and s1[i] == s0[i]  # own label never reaches own encoding

    changed = base.copy()
    changed.loc[cell & (folds != folds[i]), "ppm"] += 5.0
    w2, _ = price_ref.oof(changed, folds, min_rows=20, m=20.0)
    assert w2[i] != w0[i]  # labels from other folds do reach it


def test_sparse_cell_falls_back_to_ptype_then_global():
    df = _listings()
    base = price_ref.prepare(df)
    t = price_ref.fit(base, min_rows=10_000, m=20.0)  # no cell is dense enough
    ward, _ = price_ref.apply(base.head(5), t)
    assert np.allclose(ward, base.head(5)["ptype"].map(t.ptype).to_numpy())
    unseen = pd.DataFrame({"ward": ["w1"], "ptype": ["Lạ"], "street": [None]})
    assert price_ref.apply(unseen, t)[0][0] == t.glob


def test_street_effect_is_shrunk_and_unseen_street_equals_ward_ref():
    df = _listings(2000)
    df.loc[df["street_name"] == "Lê Lợi", "price"] *= np.e  # +1.0 on log ppm
    base = price_ref.prepare(df)
    t = price_ref.fit(base, min_rows=20, m=20.0)
    q = pd.DataFrame({"ward": ["w1", "w1"], "ptype": ["Nhà", "Nhà"], "street": ["le loi", "khong co"]})
    ward, street = price_ref.apply(q, t)
    assert 0.4 < street[0] - ward[0] < 1.0               # pulled up, but shrunk below the raw +1.0
    assert street[1] == ward[1]


def test_matrix_columns_follow_schema_order_and_unseen_ward_is_nan():
    df = _listings(10).assign(
        bedroom_count=2.0, bathroom_count=1.0, floor_count=3.0, frontage_width=4.0, road_width=6.0,
        house_direction="Đông - Nam", project_name=[None, "X"] * 5,
    )
    codes = {"ptype_group": normalize.code_table(normalize.PTYPES),
             "house_direction": normalize.code_table(normalize.DIRECTIONS),
             "ward_code_db": {"w1": 0, "w2": 1}}  # w3 unseen
    x = schema.build_matrix(df, np.zeros(10), np.zeros(10), codes)
    assert list(x.columns) == schema.FEATURES
    assert x.loc[df["ward_code_db"] == "w3", "ward_code_db"].isna().all()
    assert set(x["house_direction"]) == {codes["house_direction"]["dong_nam"]}
    assert x["has_project"].tolist() == [0.0, 1.0] * 5
