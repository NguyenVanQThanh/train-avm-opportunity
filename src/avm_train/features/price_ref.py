"""ward_price_ref / street_price_ref target encodings.

Computed per fold from that fold train part only (no leakage); sparse cells back off to
ptype_group-wide median then global mean. Final lookup tables ship in feature_schema.json.
"""
