"""
File:        config.py
Description: Run config loading, per-stage directory layout and the per-run config snapshot.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

from __future__ import annotations

import datetime
import hashlib
import pathlib

import yaml

DATA_ROOT = pathlib.Path("data")

_REQUIRED = {
    "run_id": str,
    "seed": int,
    "sources": dict,
    "scope": dict,
    "crosswalk": dict,
    "cleaning": dict,
    "features": dict,
    "split": dict,
    "model": dict,
    "cqr": dict,
    "evaluate": dict,
    "gate": dict,
}
_SPLIT_RANGES = ("train", "val", "test_a")


class GateError(RuntimeError):
    """A hard stop: pinned input missing or a quality gate failed. CLI maps it to exit code 2."""


def load(path: str | pathlib.Path) -> dict:
    """Loads and validates a run config; raises ValueError naming the first problem."""
    cfg = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    for key, typ in _REQUIRED.items():
        if key not in cfg:
            raise ValueError(f"config missing key: {key}")
        if not isinstance(cfg[key], typ):
            raise ValueError(f"config key {key} must be {typ.__name__}")
    for name in _SPLIT_RANGES:
        try:
            lo, hi = (datetime.date.fromisoformat(d) for d in cfg["split"][name])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"split.{name} must be [start, end] ISO dates") from exc
        if lo > hi:
            raise ValueError(f"split.{name} start is after end")
    return cfg


def canonical(cfg: dict) -> str:
    """Key-sorted YAML text: the exact form stored in the snapshot and hashed."""
    return yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True)


_EVAL_ONLY = ("evaluate", "gate")


def sha256(cfg: dict) -> str:
    """Hash of the settings that shape the trained model, recorded in every artifact.

    Evaluation knobs and gate thresholds are excluded: changing them must not invalidate a model.
    """
    return hashlib.sha256(
        canonical({k: v for k, v in cfg.items() if k not in _EVAL_ONLY}).encode("utf-8")
    ).hexdigest()


def snapshot(cfg: dict) -> pathlib.Path:
    """Writes data/<run_id>/config.yaml on first use; refuses a run_id whose config has changed.

    Stages of one run must share one config. A different setting is a different version: give it
    a new run_id (or delete the snapshot knowingly while still iterating on an unreleased run).
    """
    dest = DATA_ROOT / cfg["run_id"] / "config.yaml"
    text = canonical(cfg)
    if dest.exists() and dest.read_text(encoding="utf-8") != text:
        raise GateError(
            f"config changed for run_id '{cfg['run_id']}' since its snapshot ({dest}); "
            "use a new run_id for a new version"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return dest


def stage_dir(cfg: dict, stage: str) -> pathlib.Path:
    """Returns data/<run_id>/<stage>, creating it."""
    d = DATA_ROOT / cfg["run_id"] / stage
    d.mkdir(parents=True, exist_ok=True)
    return d
