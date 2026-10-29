"""
File:        cli.py
Description: CLI entry: `avm-train <stage> --config configs/<run>.yaml`.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude

Each stage reads the previous stage's output under data/<run_id>/ and writes its own.
A stage module exposes `run(cfg, args)`; GateError means a hard stop (exit 2).
"""

from __future__ import annotations

import argparse
import importlib
import sys

from avm_train import config

STAGES = {
    "fetch": "avm_train.extract.hf",
    "crosswalk": "avm_train.extract.crosswalk",
    "clean": "avm_train.clean.rules",
    "split": "avm_train.split",
    "features": "avm_train.features.schema",
    "baseline": "avm_train.train.baseline",
    "train": "avm_train.train.lgbm",
    "evaluate": "avm_train.evaluate.gate",
    "export": "avm_train.export.artifacts",
}


def main(argv: list[str] | None = None) -> int:
    # Vietnamese text in logs must not crash on the Windows cp1252 console.
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="avm-train")
    p.add_argument("stage", choices=list(STAGES))
    p.add_argument("--config", required=True)
    p.add_argument("--wards-file", help="CSV export of V-Nexus wards.code (crosswalk DB check)")
    p.add_argument("--force-draft", action="store_true",
                   help="export even when the evaluate gate failed; the manifest is flagged DRAFT")
    args = p.parse_args(argv)

    cfg = config.load(args.config)
    run = getattr(importlib.import_module(STAGES[args.stage]), "run", None)
    if run is None:
        print(f"stage '{args.stage}' not implemented yet", file=sys.stderr)
        return 1
    try:
        config.snapshot(cfg)
        run(cfg, args)
    except config.GateError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
