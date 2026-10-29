"""
File:        probe_hf.py
Description: Probe the Hugging Face dataset repo for revisions and shard files. Read-only.

Created at:  2026-10-28
Created by:  claude
Updated at:  2026-10-29
Updated by:  claude
"""

from huggingface_hub import HfApi

api = HfApi()
for repo in ["tinixai/vietnam-real-estates", "tranngocminhhieu/vietnamadminunits"]:
    for rt in ["dataset", "model"]:
        try:
            info = api.repo_info(repo, repo_type=rt, files_metadata=True)
        except Exception as exc:
            print(repo, rt, "ERR", type(exc).__name__, str(exc)[:100])
            continue
        print(repo, rt, info.sha, info.last_modified)
        for f in info.siblings[:60]:
            print("  ", f.rfilename, f.size)
        break
