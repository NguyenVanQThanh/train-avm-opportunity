"""Legacy (province, district, ward) -> ward_code_new (5 digits) via admin-unit CSV.

Emits ward_confidence in {exact, default_split}, drops unmatched rows, writes
crosswalk_report.json, and aborts when match rate < crosswalk.min_match_rate.
"""
