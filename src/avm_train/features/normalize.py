"""
File:        normalize.py
Description: Shared normalizers: house_direction (8 + null), ptype_group codes, street key, category encoding.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude

Single implementation used by training AND by the parity fixtures for the Java side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from avm_train.clean.rules import PTYPE_GROUPS
from avm_train.extract.crosswalk import norm

# Order defines the integer codes shipped in feature_schema.json; never reorder, only append.
DIRECTIONS = ("bac", "dong", "dong_bac", "dong_nam", "nam", "tay", "tay_bac", "tay_nam")
PTYPES = tuple(sorted(PTYPE_GROUPS))


def normalize_direction(value) -> str | None:
    """Maps 'Đông Nam' / 'Đông - Nam' / 'đông nam' to 'dong_nam'; anything else is None."""
    key = norm(value)
    if key is None:
        return None
    key = key.replace(" ", "_")
    return key if key in DIRECTIONS else None


def street_key(value) -> str | None:
    """Accent-stripped, lowercase street name; None when absent. Raw text never enters features."""
    return norm(value)


def encode(values: pd.Series, table: dict[str, int]) -> np.ndarray:
    """Integer category codes as float64; values missing from the table become NaN."""
    return values.map(table).astype("float64").to_numpy()


def code_table(names) -> dict[str, int]:
    """Stable name -> code table in the given order."""
    return {n: i for i, n in enumerate(names)}
