"""
File:        probe_gh.py
Description: Probe the GitHub API for the admin-unit crosswalk repo (commits, file listing). Read-only.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

import json
import urllib.request

REPO = "tranngocminhhieu/vietnamadminunits"


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "probe"})) as r:
        return json.load(r)


meta = get(f"https://api.github.com/repos/{REPO}")
print("repo", meta["full_name"], (meta.get("license") or {}).get("spdx_id"), meta["default_branch"], meta["pushed_at"])
sha = get(f"https://api.github.com/repos/{REPO}/commits/{meta['default_branch']}")["sha"]
print("head", sha)
tree = get(f"https://api.github.com/repos/{REPO}/git/trees/{sha}?recursive=1")
for t in tree["tree"]:
    if t["type"] == "blob" and t["path"].lower().endswith((".csv", ".parquet", ".json")):
        print("file", t["path"], t.get("size"))
