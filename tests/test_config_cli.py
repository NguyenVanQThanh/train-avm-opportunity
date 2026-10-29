"""
File:        test_config_cli.py
Description: Tests for config loading, validation and CLI stage dispatch.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

import pathlib

import pytest
import yaml

from avm_train import cli, config
from avm_train.extract import hf

CFG = pathlib.Path(__file__).parents[1] / "configs" / "v0_hf.yaml"


def _cfg_with(tmp_path, mutate):
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    mutate(cfg)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_shipped_config_is_valid():
    assert config.load(CFG)["run_id"] == "v0-hf"


def test_missing_key_rejected(tmp_path):
    p = _cfg_with(tmp_path, lambda c: c.pop("gate"))
    with pytest.raises(ValueError, match="gate"):
        config.load(p)


def test_inverted_date_range_rejected(tmp_path):
    p = _cfg_with(tmp_path, lambda c: c["split"].update(val=["2026-03-01", "2026-02-01"]))
    with pytest.raises(ValueError, match="split.val"):
        config.load(p)


def test_unknown_stage_exits_nonzero():
    with pytest.raises(SystemExit) as e:
        cli.main(["nope", "--config", str(CFG)])
    assert e.value.code != 0


@pytest.mark.parametrize("key", ["hf_revision", "crosswalk_commit"])
def test_fetch_refuses_unpinned(key):
    cfg = config.load(CFG)
    cfg["sources"][key] = None
    with pytest.raises(config.GateError, match=key):
        hf.run(cfg)


def test_stage_without_run_returns_1(monkeypatch):
    monkeypatch.setitem(cli.STAGES, "export", "avm_train.export")
    assert cli.main(["export", "--config", str(CFG)]) == 1
