#!/usr/bin/env python3
"""Metadata-only crawl: chaptered 'no talking' ASMR videos -> ranked URL list + CSV. Needs cookies on datacenter IPs."""
import json, os, re, sys, time
from collections import Counter
import yt_dlp
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
BASE = ["no talking", "no talking triggers", "no talking tapping", "no talking scratching", "no talking crinkles", "no talking brushing", "no talking mouth sounds", "no talking mic scratching",
        "no talking fast triggers", "no talking slow triggers", "no talking sleep", "no talking wood", "no talking glass", "no talking plastic", "no talking sticky", "no talking fabric", "no talking water",
        "no talking page turning", "no talking hand sounds", "no talking mic brushing", "no talking tapping timestamps", "no talking trigger assortment", "no talking 1 hour", "no talking 2 hours",
        "no talking ear to ear", "no talking lofi", "no talking cutting", "no talking paper", "no talking typing", "no talking eating", "no talking licking", "no talking kisses", "no talking breathing",
        "no talking ear cleaning", "no talking ear cupping", "no talking hair brushing", "no talking foam", "no talking wooden", "no talking metal", "no talking box tapping", "no talking nail tapping",
        "no talking gentle", "no talking aggressive", "no talking chaotic", "no talking unpredictable", "no talking tingly", "no talking rain", "no talking crisp", "no talking layered", "no talking compilation"]
QUERIES = [f"asmr {q}" for q in BASE]
opts = {"quiet": True, "skip_download": True, "extract_flat": False, "noplaylist": True, "ignoreerrors": True, "no_warnings": True, "sleep_interval_requests": 1.0}
if os.path.exists("/root/t2a/cookies.txt"): opts["cookiefile"] = "/root/t2a/cookies.txt"
ydl = yt_dlp.YoutubeDL(opts); seen = {}; t0 = time.time()
for q in QUERIES:
    try: res = ydl.extract_info(f"ytsearch{N}:{q}", download=False)
    except Exception as e: print("search fail", q, str(e)[:80], flush=True); continue
    for e in (res or {}).get("entries") or []:
        if not e or e.get("id") in seen: continue
        title = (e.get("title") or ""); nt = bool(re.search(r"no[\s-]?talk", title, re.I))
        seen[e["id"]] = {"id": e["id"], "title": title, "no_talking_title": nt, "duration": e.get("duration"), "channel": e.get("channel") or e.get("uploader"), "views": e.get("view_count"),
                         "chapters": [{"t": c.get("start_time"), "end": c.get("end_time"), "title": c.get("title")} for c in (e.get("chapters") or [])]}
    print(f"{q!r}: unique {len(seen)} ({time.time()-t0:.0f}s)", flush=True)
json.dump(list(seen.values()), open("yt_scope_notalking.json", "w"))
vids = [v for v in seen.values() if v["no_talking_title"] and len(v["chapters"]) >= 3]
print(f"\nvideos={len(seen)} no-talking-titled with>=3 chapters={len(vids)} hours={sum((v['duration'] or 0) for v in vids)/3600:.0f}")
KEYS = {"tapping": r"tap", "scratching": r"scratch", "crinkles": r"crinkl", "brushing": r"brush", "mouth sounds": r"mouth", "licking": r"lick", "kissing": r"kiss", "breathing": r"breath", "page turning": r"page|book", "liquid": r"water|liquid|pour|spray", "fabric": r"fabric|cloth", "sticky": r"sticky|slime", "cutting": r"cut|scissor", "mic": r"mic ", "hand sounds": r"hand", "wood": r"wood", "glass": r"glass", "plastic": r"plastic", "typing": r"typ", "writing": r"writ"}
secs = Counter()
for v in vids:
    for c in v["chapters"]:
        t = (c["title"] or "").lower(); d = (c["end"] or 0) - (c["t"] or 0)
        for k, p in KEYS.items():
            if re.search(p, t): secs[k] += d
print("class hours:", {k: round(s/3600, 1) for k, s in secs.most_common()})
vids.sort(key=lambda v: -len(v["chapters"]))
with open("yt_notalking_urls.txt", "w") as f:
    for v in vids: f.write(f"https://www.youtube.com/watch?v={v['id']}\n")
print("wrote yt_notalking_urls.txt", len(vids))
