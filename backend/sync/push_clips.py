#!/usr/bin/env python3
"""Queue clips on the crowd-labelling site (Ear Check): upload the audio under an opaque name, register the clip.

Which clips are worth a listener's time (from fuse_labels.py output):
  * unsure      -- some class's fused P is in [--lo, --hi]: the machines and people disagree
  * unlabelled  -- clips nobody has labelled yet (e.g. probe_fused.py's mined windows), from --unlabelled
  * control     -- clips people already labelled, mixed in at --control-frac so each listener's reliability can be
                   measured; on the site they look exactly like the others
Priority orders the queue: unsure clips by how unsure, the rest spread across the same range so controls cannot be
told apart by when they arrive. Audio files are named by a random UUID, so nothing about a clip's source leaks.

  export SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=...      # service role: keep it off the browser and out of git
  python push_clips.py --fused fused.jsonl --clips D:/t2a/pool D:/t2a/pool_yt --humans judges/human/*.jsonl \\
      --n-unsure 2000 --unlabelled mine.jsonl --n-unlabelled 1000 --batch 2026-10-a
  python push_clips.py --manifest manifest.jsonl --batch local-test        # {uid, path, kind?} rows, no selection
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def env(name: str) -> str:
    v = os.environ.get(name)
    if not v: sys.exit(f"set {name}")
    return v.rstrip("/")


class Site:
    def __init__(self):
        self.url, key = env("SUPABASE_URL"), env("SUPABASE_SERVICE_ROLE_KEY")
        self.h = {"apikey": key, "Authorization": f"Bearer {key}"}

    def upload(self, bucket: str, name: str, data: bytes, ctype: str = "audio/mpeg") -> None:
        r = requests.post(f"{self.url}/storage/v1/object/{bucket}/{name}", data=data, timeout=60,
                          headers={**self.h, "Content-Type": ctype, "x-upsert": "true"})
        r.raise_for_status()

    def rpc(self, fn: str, **args):
        r = requests.post(f"{self.url}/rest/v1/rpc/{fn}", json=args, timeout=120, headers=self.h)
        if not r.ok: sys.exit(f"{fn}: {r.status_code} {r.text}")
        return r.json()


def sanitize(uid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", uid)


def find_audio(dirs: list[Path]) -> dict[str, Path]:
    """Pool clips are saved as <sanitized uid>.mp3 under <pool>/pool_*/clips (label_pool.py)."""
    out = {}
    for d in dirs:
        for f in list(d.glob("pool_*/clips/*.mp3")) + list(d.glob("*.mp3")):
            out.setdefault(f.stem, f)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", type=Path)
    ap.add_argument("--clips", type=Path, nargs="*", default=[], help="dirs holding the pool clips")
    ap.add_argument("--humans", type=Path, nargs="*", default=[], help="people's label files: their clips become controls")
    ap.add_argument("--unlabelled", type=Path, default=None, help="jsonl of uids nobody labelled yet (probe_fused.py mine.jsonl)")
    ap.add_argument("--manifest", type=Path, default=None, help="jsonl {uid, path, kind?}: queue exactly these")
    ap.add_argument("--n-unsure", type=int, default=1000)
    ap.add_argument("--n-unlabelled", type=int, default=0)
    ap.add_argument("--control-frac", type=float, default=0.1)
    ap.add_argument("--lo", type=float, default=0.2)
    ap.add_argument("--hi", type=float, default=0.8)
    ap.add_argument("--target-votes", type=int, default=3)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rng = random.Random(a.batch)
    picks: list[dict] = []                                    # {uid, path, kind, priority, info}
    if a.manifest:
        for l in open(a.manifest, encoding="utf-8"):
            r = json.loads(l)
            picks.append({"uid": r["uid"], "path": Path(r["path"]), "kind": r.get("kind", "unlabelled"),
                          "priority": float(r.get("priority", rng.random())), "info": r.get("info", {})})
    else:
        audio = find_audio(a.clips)
        F = {json.loads(l)["uid"]: json.loads(l)["probs"] for l in open(a.fused, encoding="utf-8")} if a.fused else {}
        people = {}
        for f in a.humans:
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                if r.get("uid") and not r.get("skipped"): people.setdefault(r["uid"], {})[r.get("who") or f.stem] = r["labels"]
        have = lambda u: sanitize(u) in audio
        # how unsure: the class closest to a coin flip (1 = some class at exactly 0.5)
        unsure = sorted(((1 - 2 * min(abs(p - 0.5) for p in probs.values()), u) for u, probs in F.items()
                         if u not in people and have(u) and any(a.lo <= p <= a.hi for p in probs.values())), reverse=True)
        for score, u in unsure[: a.n_unsure]:
            picks.append({"uid": u, "path": audio[sanitize(u)], "kind": "unsure", "priority": score,
                          "info": {"fused": {k: v for k, v in F[u].items() if v >= 0.1}}})
        if a.unlabelled:
            mined = [json.loads(l) for l in open(a.unlabelled, encoding="utf-8")]
            mined = [m for m in mined if have(m["uid"]) and m["uid"] not in F][: a.n_unlabelled]
            for m in mined:
                picks.append({"uid": m["uid"], "path": audio[sanitize(m["uid"])], "kind": "unlabelled",
                              "priority": rng.uniform(0.3, 1.0), "info": {k: v for k, v in m.items() if k != "uid"}})
            if len(mined) < a.n_unlabelled: print(f"only {len(mined)} unlabelled clips have audio on disk (cut them first)")
        ctrl = [u for u in people if have(u)]
        rng.shuffle(ctrl)
        for u in ctrl[: int(len(picks) * a.control_frac)]:
            picks.append({"uid": u, "path": audio[sanitize(u)], "kind": "control", "priority": rng.uniform(0.3, 1.0),
                          "info": {"people": people[u]}})
    kinds = {k: sum(p["kind"] == k for p in picks) for k in ("unsure", "unlabelled", "control")}
    print(f"{len(picks)} clips: {kinds}")
    if a.dry_run or not picks: return 0
    site = Site()
    known = set()
    for i in range(0, len(picks), 500):
        known |= set(site.rpc("admin_known_sources", p_source_uids=[p["uid"] for p in picks[i:i + 500]]))

    def up(p: dict) -> dict:
        name = None                                           # already queued: refresh priority, keep its audio
        if p["uid"] not in known:
            name = f"{uuid.uuid4()}.mp3"
            site.upload("clips", name, p["path"].read_bytes())
        return {"audio_path": name, "source_uid": p["uid"], "kind": p["kind"], "priority": round(p["priority"], 4),
                "target_votes": a.target_votes, "batch": a.batch, "info": p["info"]}

    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(up, picks))
    added = 0
    for i in range(0, len(rows), 500):
        added += site.rpc("admin_add_clips", p_clips=rows[i:i + 500])
    print(f"queued {added} new clips (batch {a.batch}); already queued: {len(rows) - added} refreshed")
    print(json.dumps(site.rpc("admin_queue_stats"), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
