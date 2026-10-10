#!/usr/bin/env python3
"""Find YouTube "no talking" trigger videos for the classes people have labelled least, by title, tags and chapters.

For each target class: flat yt-dlp searches (one request per query), then full metadata only for results whose
title says no talking. A video becomes a candidate when
  * its title names exactly one trigger class (single: the whole video, minus intro/outro, is that class), or the
    title is generic and its tags name exactly one class (tags are where creators list what the video is), or
  * it has chapters whose titles each name exactly one class: those chapters (>= 60 s) are kept with their class.
Assortment titles ("tapping, scratching, brushing") without chapters are skipped: no window could be trusted.
The content filter (content_filter.py) drops blocked titles. Output rows are yt_candidates.jsonl rows; merge them with
merge_candidates.py --have (done + queued + rejected ids) and append to the fetcher's candidate list.

Metadata only (no downloads); polite: one request at a time with a pause between.

  python discover_yt_triggers.py --classes spraying "paper rustling" crinkling --per-query 40 \\
      --have D:/t2a/yt_dense/done.txt D:/t2a/yt_dense/yt_candidates.jsonl --out D:/t2a/yt_dense/discover/group_1008.jsonl
"""
from __future__ import annotations
import argparse, json, re, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content_filter import blocked_title

# class -> title / chapter / tag pattern; the most specific class wins where two match (mic scratching is mic touching)
PAT = {
    "spraying": r"spray|spritz|misting|aerosol",
    "paper rustling": r"\bpapers?\b|newspaper|cardboard|envelope",
    "page turning": r"page ?turn|turning pages|page flip|flipping (?:through )?pages|book sounds|\bbooks?\b|magazine",
    "crinkling": r"crinkl|plastic wrap|cling ?film|wrappers?\b|bubble ?wrap|\bfoil\b|crispy plastic",
    "tapping": r"tapping|\btaps\b",
    "brushing": r"brush",
    "cutting": r"cutting|scissors?|slicing|chopping|\bknife\b|carving",
    "liquid": r"water|liquid|pouring|dripping|\bdrips\b|splash",
    "sticky": r"sticky|\btape\b|slime|\bglue\b|honey|velcro",
    "scratching": r"scratch",
    "microphone touching": r"mic(?:rophone)? (?:scratch|touch|rub|tap|pump|swirl)|fluffy mic|mic cover|mic foam",
    "fabric rustling": r"fabric|cloth|clothing|jacket|blanket|corduroy|leather|silk",
    "typing": r"typing|keyboard",
    "writing": r"writing|pencil|\bpen\b|calligraphy",
}
OVERRIDES = {"microphone touching": {"scratching", "tapping", "brushing"}, "crinkling": {"paper rustling"},
             "page turning": {"paper rustling"}}
NO_TALK = re.compile(r"no[\s-]*talk|no[\s-]*speaking|without talking", re.I)
QUERY_TERMS = {
    "spraying": ["spray bottle", "spraying", "spray sounds", "hair spray", "mist spray", "water spray", "spritzing", "hairspray", "aerosol can", "spray mist sounds"],
    "paper rustling": ["paper sounds", "paper crumpling", "newspaper", "paper rustling", "cardboard", "tissue paper", "crumpling paper", "envelope sounds", "paper tearing", "cardboard sounds"],
    "page turning": ["page turning", "book sounds", "flipping pages", "page flipping", "book page turning", "old book sounds", "flipping through book", "magazine page turning"],
    "crinkling": ["crinkle", "crinkling", "plastic crinkles", "bubble wrap", "plastic wrap"],
    "tapping": ["tapping", "fast tapping", "nail tapping", "wood tapping", "glass tapping"],
    "brushing": ["brushing", "mic brushing", "brush sounds", "makeup brushes", "fluffy brushes", "brushing the camera", "fluffy brush", "hair brushing", "paint brush sounds"],
    "cutting": ["cutting", "scissors", "cutting sounds", "soap cutting", "chopping", "kinetic sand cutting", "soap carving", "foam cutting", "scissors cutting paper", "cutting vegetables"],
    "liquid": ["water sounds", "liquid sounds", "pouring water", "water bottle", "water drops", "liquid pouring", "drinking water sounds", "water bubbles", "water gel"],
    "sticky": ["sticky", "tape sounds", "sticky fingers", "slime", "sticky tape", "duct tape", "sticky hands", "honey sounds", "tape peeling", "velcro"],
}


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)


def classes_in(text: str) -> set[str]:
    t = (text or "").lower()
    hit = {c for c, p in PAT.items() if re.search(p, t)}
    for winner, losers in OVERRIDES.items():
        if winner in hit: hit -= losers
    return hit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=list(QUERY_TERMS))
    ap.add_argument("--per-query", type=int, default=40, help="search results per query")
    ap.add_argument("--have", type=Path, nargs="*", default=[], help="ids already fetched / queued / rejected (txt or jsonl)")
    ap.add_argument("--min-min", type=float, default=5.0)
    ap.add_argument("--max-min", type=float, default=240.0)
    ap.add_argument("--cookies", type=Path, default=None)
    ap.add_argument("--pause", type=float, default=5.0, help="seconds between full-metadata requests")
    ap.add_argument("--fail-streak", type=int, default=5, help="this many unavailable videos in a row = rate limited: back off")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import yt_dlp
    have = set()
    for f in a.have:
        if not f.exists(): continue
        for l in open(f, encoding="utf-8"):
            l = l.strip()
            if not l: continue
            have.add(json.loads(l).get("id") if l.startswith("{") else l)
    base = {"quiet": True, "no_warnings": True, "ignoreerrors": True, "sleep_interval_requests": 1.0}
    if a.cookies and a.cookies.exists(): base["cookiefile"] = str(a.cookies)
    flat, full = yt_dlp.YoutubeDL({**base, "extract_flat": True}), yt_dlp.YoutubeDL({**base, "skip_download": True})
    seen, out, why, streak = set(have), [], Counter(), [0]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(a.out, "a", encoding="utf-8")
    for cls in a.classes:
        n_cls = 0
        for term in QUERY_TERMS.get(cls, [cls]):
            res = flat.extract_info(f"ytsearch{a.per_query}:asmr {term} no talking", download=False) or {}
            for e in res.get("entries") or []:
                if not e or not e.get("id") or e["id"] in seen: continue
                seen.add(e["id"]); title = e.get("title") or ""
                dur = float(e.get("duration") or 0)
                if not NO_TALK.search(title): why["talking title"] += 1; continue
                if blocked_title(title): why["content filter"] += 1; continue
                if dur and not (a.min_min * 60 <= dur <= a.max_min * 60): why["length"] += 1; continue
                time.sleep(a.pause)
                info = full.extract_info(f"https://www.youtube.com/watch?v={e['id']}", download=False)
                if not info:
                    # YouTube answers a rate-limited session with "unavailable" for every video: a run of them means
                    # back off (an hour, as its message says), not that the videos are gone
                    why["unavailable"] += 1; streak[0] += 1; seen.discard(e["id"])
                    if streak[0] >= a.fail_streak:
                        log(f"{streak[0]} unavailable in a row: rate limited, backing off 60 min"); time.sleep(3600); streak[0] = 0
                    continue
                streak[0] = 0
                dur = float(info.get("duration") or 0)
                row = {"id": info["id"], "url": f"https://www.youtube.com/watch?v={info['id']}", "title": title,
                       "channel": info.get("channel") or info.get("uploader"), "channel_id": info.get("channel_id"),
                       "duration_s": int(dur), "no_talking": True}
                chapters = []
                for ch in info.get("chapters") or []:
                    hit = classes_in(ch.get("title"))
                    if len(hit) == 1 and (ch.get("end_time") or 0) - (ch.get("start_time") or 0) >= 60:
                        chapters.append({"start": float(ch["start_time"]), "end": float(ch["end_time"]),
                                         "title": ch.get("title"), "cls": hit.pop()})
                in_title = classes_in(title)
                in_tags = classes_in(" ".join(info.get("tags") or []))
                if len(chapters) >= 2:
                    top = Counter(c["cls"] for c in chapters).most_common(1)[0][0]
                    row.update(cls=top, kind="chapter", chapters=chapters,
                               evidence=f"{len(chapters)} classified chapters: " + ", ".join(sorted({c['cls'] for c in chapters})))
                elif len(in_title) == 1:
                    row.update(cls=in_title.pop(), kind="single", chapters=[], evidence=title)
                elif not in_title and len(in_tags) == 1:
                    row.update(cls=in_tags.pop(), kind="single", chapters=[], evidence="tags: " + ", ".join((info.get("tags") or [])[:12]))
                else:
                    why["assortment / unclear" if in_title or in_tags else "no class named"] += 1; continue
                fh.write(json.dumps(row) + "\n"); fh.flush(); out.append(row); n_cls += 1
            log(f"{cls:16} {term!r:22} candidates so far {n_cls}")
        log(f"== {cls}: {n_cls} candidates")
    fh.close()
    log(f"DISCOVER_DONE {len(out)} candidates " + str(dict(Counter(r['cls'] for r in out))) + f"; skipped {dict(why)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
