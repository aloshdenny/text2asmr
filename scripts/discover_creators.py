#!/usr/bin/env python3
"""Discover NEW soundgasm creators rich in deficit labels via the ilovesoundgasm search API. Network-only (run on the DO box).
Output: expansion_plan.jsonl (one creator per line: uploader, route repo, matched posts, minutes, GB estimate, tag hits)."""
from __future__ import annotations
import argparse, json, re, time, urllib.parse, urllib.request
from collections import Counter, defaultdict
QUERIES = {
    "kissing": ["kissing", "kisses", "kiss sounds", "sloppy kissing", "mwah", "smooches", "cheek kisses", "neck kisses", "makeout", "making out"],
    "moaning": ["moaning", "moans", "whimpering", "heavy moaning", "moaning only", "orgasm sounds", "whimpers"],
    "mouth sounds": ["mouth sounds", "wet sounds", "licking", "ear licking", "lip smacking", "sloppy", "slurping", "tongue"],
    "breathing": ["heavy breathing", "breathing", "panting", "breathy"],
    # breadth: gender/genre queries (deficit tags only lift yield ~1.3x, so NEW creators are what scales the corpus)
    "_generic_f": ["F4M", "F4A", "F4F", "girlfriend", "gfe", "mommy", "wife", "older woman", "comfort", "sleep aid", "cuddles", "roleplay", "script fill", "asmr", "whisper", "gentle fdom", "praise", "teasing"],
    "_generic_m": ["M4F", "M4A", "M4M", "boyfriend", "bfe", "daddy", "husband", "older man", "male moaning", "mdom", "gentle mdom", "comfort m4f", "sleep aid m4f", "roleplay m4f", "script fill m4f", "whisper m4f", "praise m4f", "aftercare m4f"],
}
TAGPAT = {"kissing": r"kiss|mwah|smooch|makeout|making out", "moaning": r"moan|whimper|orgasm", "mouth sounds": r"mouth|wet sound|lick|slurp|sloppy|tongue|smack", "breathing": r"breath|pant"}
def route(cat: str, title: str):
    c = (cat or "").lower(); t = (title or "").upper()
    m = re.search(r"\b([FM])[FM]?4", t) or re.search(r"^([fm])", c)
    if not m: return None
    return "aoxo/audios2" if m.group(1).upper() == "F" else "aoxo/audios3"
def search(q, pages, sleep=0.6):
    cursor = None
    for _ in range(pages):
        url = "https://ilovesoundgasm.com/api/search?" + urllib.parse.urlencode({"q": q, **({"cursor": cursor} if cursor else {})})
        for i in range(4):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as r: d = json.load(r); break
            except Exception:
                if i == 3: return
                time.sleep(5 * (i + 1))
        for it in d.get("items", []): yield it
        cursor = d.get("nextCursor")
        if not d.get("hasMore") or not cursor: return
        time.sleep(sleep)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exclude", required=True); ap.add_argument("--pages", type=int, default=150); ap.add_argument("--out", default="expansion_plan.jsonl")
    ap.add_argument("--min-posts", type=int, default=0); ap.add_argument("--min-seen", type=int, default=3); ap.add_argument("--cap-files", type=int, default=150); ap.add_argument("--mb-per-min", type=float, default=0.75)
    a = ap.parse_args(); excluded = {l.split("\t")[0].strip().lower() for l in open(a.exclude) if l.strip()}
    posts = {}; t0 = time.time()
    for label, qs in QUERIES.items():
        for q in qs:
            n = 0
            for it in search(q, a.pages):
                posts[it["id"]] = it; n += 1
            print(f"[{time.strftime('%H:%M:%S')}] {label:13s} {q!r:20s} +{n:5d} posts (total {len(posts)}, {time.time()-t0:.0f}s)", flush=True)
    by_up = defaultdict(list)
    for it in posts.values(): by_up[it["uploader"]].append(it)
    plan = []; skipped_existing = 0
    for up, its in by_up.items():
        if up.lower() in excluded: skipped_existing += 1; continue
        hits = Counter(); routes = Counter(); mins = 0
        for it in its:
            text = " ".join([it.get("title", "")] + (it.get("tags") or [])).lower()
            for lab, pat in TAGPAT.items():
                if re.search(pat, text): hits[lab] += 1
            r = route(it.get("category"), it.get("title")); routes[r or "?"] += 1; mins += (it.get("duration") or 0)
        matched = sum(hits.values())
        if matched < a.min_posts or len(its) < a.min_seen: continue
        repo = routes.most_common(1)[0][0]
        if repo == "?": continue
        take = min(len(its), a.cap_files); est_gb = mins / max(1, len(its)) * take * a.mb_per_min / 1024
        plan.append({"uploader": up, "repo": repo, "posts_seen": len(its), "matched_posts": matched, "hits": dict(hits), "minutes_seen": mins, "take_files": take, "est_gb": round(est_gb, 2),
                     "score": hits["kissing"] * 3 + hits["moaning"] * 2 + hits["mouth sounds"] * 1.5 + hits["breathing"], "sample_urls": [it["url"] for it in its[:3]]})
    plan.sort(key=lambda p: -p["score"])
    with open(a.out, "w") as f:
        for p in plan: f.write(json.dumps(p) + "\n")
    tot = Counter(); gb = Counter()
    for p in plan: tot[p["repo"]] += 1; gb[p["repo"]] += p["est_gb"]
    print(f"\nposts={len(posts)} uploaders={len(by_up)} excluded_existing={skipped_existing} planned_creators={len(plan)}")
    for repo in tot: print(f"  {repo}: {tot[repo]} creators, ~{gb[repo]:.0f} GB (cap {a.cap_files} files/creator)")
    print("top 10:", [(p["uploader"], p["repo"].split('/')[-1], p["matched_posts"], p["hits"]) for p in plan[:10]])
if __name__ == "__main__": main()
