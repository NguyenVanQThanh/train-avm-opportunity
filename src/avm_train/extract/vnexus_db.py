"""
File:        vnexus_db.py
Description: Read-only access to the V-Nexus Postgres (replica or dev). Never writes.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude

Connection comes from AVM_DB_HOST/PORT/NAME/USER/PASSWORD env vars so no credential lives in
the repo or in config files.
"""

from __future__ import annotations

import os

import pandas as pd


def connect():
    import psycopg  # optional dependency: pip install -e ".[db]"

    conn = psycopg.connect(
        host=os.environ["AVM_DB_HOST"],
        port=os.environ.get("AVM_DB_PORT", "5432"),
        dbname=os.environ["AVM_DB_NAME"],
        user=os.environ["AVM_DB_USER"],
        password=os.environ["AVM_DB_PASSWORD"],
        connect_timeout=15,
    )
    conn.read_only = True
    return conn


def fetch_wards() -> pd.DataFrame:
    """Returns every ward as (code, name, province_code, province_name); code is the 8-digit wards.code."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT w.code, w.name, p.code, p.name FROM wards w JOIN provinces p ON p.id = w.province_id"
        )
        return pd.DataFrame(cur.fetchall(), columns=["code", "name", "province_code", "province_name"])
