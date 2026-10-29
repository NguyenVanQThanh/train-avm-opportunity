"""
File:        test_crosswalk.py
Description: Tests for the legacy-to-new ward crosswalk and the V-Nexus ward mapping.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

import pandas as pd
import pytest

from avm_train.config import GateError
from avm_train.extract import crosswalk as cwm


def _cw():
    rows = [
        # (province, district, ward, newWardCode, divided, default)
        ("Hồ Chí Minh", "Quận 3", "Phường 14", "27142", "False", None),
        ("Hồ Chí Minh", "Quận 3", "Phường Võ Thị Sáu", "27139", "False", None),
        ("Hồ Chí Minh", "Quận 3", "Phường 4", "27154", "True", "True"),
        ("Hồ Chí Minh", "Quận 3", "Phường 4", "27155", "True", "False"),
        ("Hồ Chí Minh", "Thành phố Thủ Đức", "Phường An Phú", "26000", "False", None),
        ("Bình Dương", "Thành phố Dĩ An", "Phường Dĩ An", "26100", "False", None),
    ]
    return pd.DataFrame(
        {
            "provinceShort": [r[0] for r in rows],
            "district": [r[1] for r in rows],
            "ward": [r[2] for r in rows],
            "newWardCode": [r[3] for r in rows],
            "newProvinceCode": "79",
            "isDividedWard": [r[4] for r in rows],
            "isDefaultNewWard": [r[5] for r in rows],
        }
    )


def _hf(*rows):
    return pd.DataFrame(rows, columns=["province_name", "district_name", "ward_name"])


@pytest.mark.parametrize(
    "raw,expected",
    [("Quận 3", "3"), ("Phường 14", "14"), ("Đông Nam", "dong nam"), ("  Võ  Thị-Sáu ", "vo thi sau"), (None, None)],
)
def test_norm(raw, expected):
    assert cwm.norm(raw) == expected


def test_numeric_district_and_ward_match_exact():
    out = cwm.apply(_hf(("Hồ Chí Minh", "3", "14")), _cw())
    assert out.loc[0, "ward_code_new"] == "27142"
    assert out.loc[0, "ward_confidence"] == "exact"


def test_divided_ward_takes_default_and_is_flagged():
    out = cwm.apply(_hf(("Hồ Chí Minh", "3", "4")), _cw())
    assert (out.loc[0, "ward_code_new"], out.loc[0, "ward_confidence"]) == ("27154", "default_split")


def test_prefix_stripped_on_second_pass():
    out = cwm.apply(_hf(("Hồ Chí Minh", "Thủ Đức", "An Phú"), ("Bình Dương", "Dĩ An", "Dĩ An")), _cw())
    assert out["ward_code_new"].tolist() == ["26000", "26100"]


def test_unmatched_and_missing_location_stay_null():
    out = cwm.apply(_hf(("Hồ Chí Minh", "3", "Không có"), ("Hồ Chí Minh", None, "14")), _cw())
    assert out["ward_code_new"].isna().all()


def _wards():
    return pd.DataFrame(
        {
            "code": ["71101001", "71101002", "71101003", "71101004", "71101005", "99999999"],
            "name": ["Phường 14", "Phường Võ Thị Sáu", "Phường 4", "Phường 4B", "Phường An Phú", "Phường 14"],
            "province_code": ["29"] * 5 + ["01"],
            "province_name": ["Tp Hồ Chí Minh"] * 5 + ["Thành phố Hà Nội"],
        }
    )


def test_map_to_db_is_one_to_one_and_keeps_accents():
    cw = pd.DataFrame(
        {
            "newWard": ["Xã Thanh An", "Xã Thạnh An"],
            "newWardCode": ["25807", "27676"],
            "newProvinceCode": "79",
        }
    )
    wards = pd.DataFrame(
        {"code": ["71113062", "70143168"], "name": ["Xã Thanh An", "Xã Thạnh An"], "province_code": "29"}
    )
    mapping, problems = cwm.map_to_db(cw, wards, "79", "29")
    assert dict(zip(mapping["ward_code_new"], mapping["ward_code_db"])) == {
        "25807": "71113062", "27676": "70143168"}
    assert problems == {"new_wards_without_exactly_one_db_match": [], "db_wards_not_targeted": []}


def test_map_to_db_reports_missing_and_untargeted():
    cw = pd.DataFrame({"newWard": ["Phường A"], "newWardCode": ["1"], "newProvinceCode": "79"})
    wards = pd.DataFrame({"code": ["7"], "name": ["Phường B"], "province_code": "29"})
    mapping, problems = cwm.map_to_db(cw, wards, "79", "29")
    assert mapping.empty
    assert problems["new_wards_without_exactly_one_db_match"] == ["Phường A"]
    assert problems["db_wards_not_targeted"] == ["7"]


def test_run_stops_below_threshold(tmp_path, monkeypatch):
    from avm_train import config
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    cfg = {"run_id": "t", "scope": {"province_code_new": "79"}, "crosswalk": {"min_match_rate": 0.95}}
    raw = tmp_path / "t" / "raw"
    raw.mkdir(parents=True)
    _cw().to_csv(raw / "crosswalk.csv", index=False)
    _hf(("Hồ Chí Minh", "3", "14"), ("Hồ Chí Minh", "3", "Sai tên")).to_parquet(raw / "listings.parquet")
    with pytest.raises(GateError, match="match rate"):
        cwm.run(cfg)
    assert (tmp_path / "t" / "crosswalk" / "crosswalk_report.json").exists()


def test_name_key_unifies_tone_placement_but_keeps_accents():
    assert cwm.name_key("Phường Chánh Phú Hoà") == cwm.name_key("Phường Chánh Phú Hòa")
    assert cwm.name_key("Xã Thanh An") != cwm.name_key("Xã Thạnh An")
