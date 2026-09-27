#!/usr/bin/env python3
"""Tar a directory and upload it to an HF dataset. Kept as a file so pod bootstraps need no nested quoting."""
import os, subprocess, sys
from huggingface_hub import HfApi
src, repo, path_in_repo = sys.argv[1], sys.argv[2], sys.argv[3]
tar = src.rstrip("/") + ".tar"
subprocess.run(["tar", "-C", os.path.dirname(src) or ".", "-cf", tar, os.path.basename(src)], check=True)
sz = os.path.getsize(tar) / 1e9
for attempt in range(5):
    try:
        HfApi().upload_file(path_or_fileobj=tar, path_in_repo=path_in_repo, repo_id=repo, repo_type="dataset",
                            commit_message=f"{path_in_repo} ({sz:.1f} GB)")
        print(f"uploaded {tar} ({sz:.1f} GB) -> {repo}/{path_in_repo}"); break
    except Exception as e:
        import time; print(f"retry {attempt}: {type(e).__name__} {str(e)[:100]}"); time.sleep(30 * (attempt + 1))
