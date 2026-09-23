#!/usr/bin/env python3
"""Stream soundgasm creators from expansion_plan.jsonl into aoxo/t2a-mommy (female) / aoxo/t2a-daddy (male). DO-box friendly:
one creator at a time, small local footprint, HF commits in batches, resumable via acquired.jsonl. Skips creators already on the Hub."""
from __future__ import annotations
import argparse, json, os, shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import soundgasm_downloader as dl
from huggingface_hub import HfApi, CommitOperationAdd
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--plan", default="expansion_plan.jsonl"); ap.add_argument("--work", type=Path, default=Path("/root/t2a/acq"))
    ap.add_argument("--max-gb", type=float, default=1000.0, help="stop after this many GB pushed (per run)"); ap.add_argument("--batch-files", type=int, default=12); ap.add_argument("--min-free-gb", type=float, default=4.0)
    ap.add_argument("--only-repo", default=""); ap.add_argument("--cap-files", type=int, default=150); a = ap.parse_args(); a.work.mkdir(parents=True, exist_ok=True)
    api = HfApi(); ledger = a.work / "acquired.jsonl"; done_creators = set(); pushed_gb = 0.0
    if ledger.exists():
        for l in open(ledger): r = json.loads(l); done_creators.add(r["uploader"]); pushed_gb += r["gb"]
    existing = {}
    for repo in ("aoxo/t2a-mommy", "aoxo/t2a-daddy"):
        existing[repo] = {f.split("/")[0].lower() for f in api.list_repo_files(repo, repo_type="dataset") if "/" in f}
    plan = [json.loads(l) for l in open(a.plan)]
    if a.only_repo: plan = [p for p in plan if p["repo"] == a.only_repo]
    log(f"plan {len(plan)} creators, {len(done_creators)} done, {pushed_gb:.1f} GB pushed so far")
    session = dl.make_session()
    for p in plan:
        up, repo = p["uploader"], p["repo"]
        if up in done_creators or up.lower() in existing[repo]: continue
        if pushed_gb >= a.max_gb: log("byte budget reached"); break
        try:
            html = dl.get_profile(session, up); pages = dl.extract_audio_pages(html, up)[: max(p["take_files"], a.cap_files)]
        except Exception as e: log(f"{up}: profile failed {type(e).__name__}"); continue
        cdir = a.work / dl.safe_filename(up); cdir.mkdir(exist_ok=True); ops = []; n_files = 0; gb = 0.0
        def flush():
            nonlocal ops
            if not ops: return
            for i in range(4):
                try: api.create_commit(repo_id=repo, repo_type="dataset", operations=ops, commit_message=f"expansion: {up} (+{len(ops)} files)"); break
                except Exception as e:
                    log(f"{up}: commit retry {i}: {str(e)[:100]}"); time.sleep(30 * (i + 1))
            for op in ops:
                try: os.remove(op.path_or_fileobj)
                except Exception: pass
            ops = []
        for page in pages:
            if shutil.disk_usage("/").free / 1e9 < a.min_free_gb: flush()
            try:
                media = dl.get_media_url(session, page)
                if not media: continue
                fname = dl.audio_filename(media); out = cdir / fname
                if out.exists(): continue
                r = session.get(media, stream=True, timeout=120); r.raise_for_status()
                with open(out, "wb") as f:
                    for chunk in r.iter_content(1 << 20): f.write(chunk)
                sz = out.stat().st_size
                if sz < 200_000: out.unlink(); continue
                ops.append(CommitOperationAdd(path_in_repo=f"{dl.safe_filename(up)}/{fname}", path_or_fileobj=str(out))); n_files += 1; gb += sz / 1e9
                if len(ops) >= a.batch_files: flush()
            except Exception as e: log(f"{up}: {type(e).__name__} on {page[-50:]}")
        flush(); shutil.rmtree(cdir, ignore_errors=True); pushed_gb += gb
        with open(ledger, "a") as f: f.write(json.dumps({"uploader": up, "repo": repo, "files": n_files, "gb": round(gb, 3), "ts": time.strftime("%F %T")}) + "\n")
        log(f"{up} -> {repo.split('/')[-1]}: {n_files} files, {gb:.2f} GB (total {pushed_gb:.1f} GB)")
    log("ACQUIRE_DONE")
if __name__ == "__main__": main()
