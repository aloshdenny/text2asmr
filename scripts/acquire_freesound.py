#!/usr/bin/env python3
"""Fill the physical-trigger classes FSD50K has no clean source for (spraying, brushing, paper rustling) and top up the
others, from Freesound (CC-licensed, human-tagged, per-clip preview download).

Like acquire_triggers.py this is OUT OF DOMAIN: Freesound clips are ordinary field/foley recordings, not close-mic
binaural ASMR. Every clip is written with source="freesound" and domain="out" so training can cap or down-weight it,
and so an in-domain evaluation never silently becomes an out-of-domain one. Clips land in a pool layout
(pool_*/clips/<uid>.mp3 + pool.jsonl) so pool_local_labels.py / label_pool.py / the crowd site read them unchanged.

Token auth (FREESOUND_API_KEY). Downloads the high-quality mp3 preview (no OAuth needed), which is all we keep anyway
(everything is recut to 6 s / 24 kHz downstream). Rate-limited politely; resumable via the manifest.

  python acquire_freesound.py --work D:/t2a/pool_freesound --per-class 400 --only spraying,brushing,paper_rustling
"""
from __future__ import annotations
import argparse, json, os, re, time, urllib.parse, urllib.request, urllib.error
from collections import Counter
from pathlib import Path

# our ontology class -> Freesound text queries (broad, several per class for coverage). Results are filtered to short,
# decently-rated clips; the class is a weak label (the search term), confirmed downstream by the judges and by people.
QUERIES = {
    "spraying":       ["spray bottle", "spray sound", "spraying", "aerosol spray", "hairspray", "pump spray", "water spray"],
    "brushing":       ["brushing sound", "makeup brush", "hair brush", "paint brush", "brush microphone", "soft brush"],
    "paper rustling": ["paper rustling", "paper crumple", "paper handling", "newspaper", "crumpling paper", "paper sheet"],
    "crinkling":      ["plastic crinkle", "crinkling", "plastic wrapper", "bubble wrap", "cellophane", "plastic bag crinkle"],
    "cutting":        ["scissors cutting", "cutting paper", "snipping", "cutting sound", "shears"],
    "sticky":         ["sticky tape", "tape peeling", "duct tape", "adhesive tape", "velcro", "sticky sound"],
    "tapping":        ["finger tapping", "tapping sound", "nail tapping", "knock tapping", "tapping wood"],
    "scratching":     ["scratching sound", "nail scratching", "scratching surface", "scratch texture"],
    "liquid":         ["water pouring", "liquid pouring", "water dripping", "pouring sound", "water trickle"],
}
SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)
def safe(uid: str) -> str: return SAFE.sub("_", uid)


def get(url: str, pause: list, timeout: int = 60) -> bytes:
    for attempt in range(6):
        wait = pause[0] - time.time()
        if wait > 0: time.sleep(wait)
        try:
            return urllib.request.urlopen(url, timeout=timeout).read()
        except urllib.error.HTTPError as e:
            if e.code == 429:                                    # throttled: back off, then retry
                pause[0] = max(pause[0], time.time() + 30 * (attempt + 1)); continue
            if e.code in (401, 404): raise
            time.sleep(5 * (attempt + 1))
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"giving up on {url[:80]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--only", default="", help="comma-separated subset of classes")
    ap.add_argument("--min-dur", type=float, default=1.5)
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--sleep", type=float, default=1.1, help="seconds between Freesound requests (free tier: 60/min)")
    a = ap.parse_args()
    token = os.environ["FREESOUND_API_KEY"]
    clips = a.work / "pool_freesound" / "clips"; clips.mkdir(parents=True, exist_ok=True)
    pool = a.work / "pool.jsonl"
    done = {json.loads(l)["uid"] for l in open(pool, encoding="utf-8")} if pool.exists() else set()
    wanted = {c.strip().replace("_", " ") for c in a.only.split(",") if c.strip()} or set(QUERIES)
    wanted = {c for c in QUERIES if c in wanted or c.replace(" ", "_") in {x.replace(" ", "_") for x in wanted}}
    pause, stats = [0.0], Counter()
    with open(pool, "a", encoding="utf-8") as pf:
        for cls in sorted(wanted):
            kept, page_seen = 0, set()
            for q in QUERIES[cls]:
                if kept >= a.per_class: break
                page = 1
                while kept < a.per_class:
                    params = {"query": q, "filter": f"duration:[{a.min_dur} TO {a.max_dur}]",
                              "fields": "id,name,license,username,duration,previews,avg_rating",
                              "page_size": 50, "page": page, "sort": "rating_desc", "token": token}
                    try:
                        d = json.loads(get("https://freesound.org/apiv2/search/text/?" + urllib.parse.urlencode(params), pause))
                    except Exception as e:
                        log(f"  {cls} {q!r} page {page}: {type(e).__name__}"); break
                    time.sleep(a.sleep)
                    results = d.get("results") or []
                    if not results: break
                    for r in results:
                        if kept >= a.per_class: break
                        uid = f"fs:{r['id']}"
                        if uid in done or uid in page_seen: continue
                        page_seen.add(uid)
                        url = (r.get("previews") or {}).get("preview-hq-mp3")
                        if not url: continue
                        try:
                            audio = get(url + ("?" if "?" not in url else "&") + "token=" + token, pause)
                        except Exception as e:
                            stats[f"{cls}:fail"] += 1; continue
                        time.sleep(a.sleep)
                        if len(audio) < 4000: stats[f"{cls}:tiny"] += 1; continue
                        (clips / f"{safe(uid)}.mp3").write_bytes(audio)
                        pf.write(json.dumps({"uid": uid, "label": cls, "src": "freesound", "repo": "freesound",
                                             "rec": uid, "start": 0.0, "fs_id": r["id"], "license": r.get("license"),
                                             "username": r.get("username"), "domain": "out"}) + "\n")
                        pf.flush(); done.add(uid); kept += 1; stats[cls] += 1
                    if not d.get("next"): break
                    page += 1
            log(f"{cls}: {kept} clips kept")
    log(f"FREESOUND_DONE {dict(stats)} -> {clips}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
