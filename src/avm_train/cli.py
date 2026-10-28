"""CLI entry: `avm-train <stage> --config configs/<run>.yaml`.

Stages run in order; each reads the previous stage's output under data/<run_id>/
and writes its own. Stage names mirror the v0 execution order.
"""

STAGES = [
    "fetch",
    "crosswalk",
    "clean",
    "split",
    "features",
    "baseline",
    "train",
    "evaluate",
    "export",
]


def main() -> None:
    raise NotImplementedError
