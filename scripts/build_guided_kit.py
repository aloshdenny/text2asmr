#!/usr/bin/env python3
"""A labelling kit aimed where human answers are worth most, with a sound guide built in.

Weak supervision can only learn which machine to trust for a class from clips a person labelled, and people had
labelled almost no physical triggers (brushing 6 positives, liquid 9, cutting ~2). Two careful listeners also
disagreed on what "crinkling" or "tapping" covers (Jaccard 0.33). So this kit:
  * opens with a sound guide: per class, the clearest examples the fusion knows (chapter title, Gemini and the
    fused probability all agree), so every listener means the same thing by each label
  * asks about the clips the fusion is least sure of (fused P between --lo and --hi for their chapter's class),
    an even number per class, listening order shuffled; the class the clip came from stays off the page

  python build_guided_kit.py --fused fused.jsonl --votes merged/ --clips yt_clips/ --n-per-class 13 --out kit.html
"""
from __future__ import annotations
import argparse, base64, json, random, re
from pathlib import Path

from build_label_html import LABEL_GROUPS

HINTS = {"tapping": "fingertips or nails tapping a surface: short, separate taps",
         "scratching": "nails dragged over a textured surface: continuous rough scraping",
         "crinkling": "plastic, foil or a wrapper squeezed: sharp crackly sounds",
         "brushing": "a brush stroked over the mic or a surface: soft swishing",
         "liquid": "water or gel: pouring, dripping, sloshing, bubbles",
         "microphone touching": "fingers on the microphone itself: muffled rubs and thumps",
         "sticky": "tacky surfaces pulling apart: tape, slime, sticky fingers",
         "fabric rustling": "cloth moving: shirts, blankets, gloves",
         "paper rustling": "paper: pages turning, crumpled paper, cardboard",
         "cutting": "scissors or a knife: snips and slices",
         "oral sounds": "licking, lip smacks, wet mouth sounds (not kisses)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", type=Path, required=True)
    ap.add_argument("--votes", type=Path, required=True, help="dir with chapter.jsonl and gemini.jsonl")
    ap.add_argument("--clips", type=Path, required=True, help="mp3 clips named by sanitized uid")
    ap.add_argument("--n-per-class", type=int, default=13)
    ap.add_argument("--guide-per-class", type=int, default=2)
    ap.add_argument("--lo", type=float, default=0.2)
    ap.add_argument("--hi", type=float, default=0.8)
    ap.add_argument("--kit-id", default="asmr-ear-check-4")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[], help="label files whose uids may not be guide examples (e.g. a listener said the example was wrong)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    F = {json.loads(l)["uid"]: json.loads(l)["probs"] for l in open(a.fused, encoding="utf-8")}
    ch = {json.loads(l)["uid"]: json.loads(l)["labels"] for l in open(a.votes / "chapter.jsonl", encoding="utf-8")}
    gem = {json.loads(l)["uid"]: set(json.loads(l)["labels"]) for l in open(a.votes / "gemini.jsonl", encoding="utf-8")}
    path = lambda u: a.clips / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', u)}.mp3"
    b64 = lambda u: "data:audio/mpeg;base64," + base64.b64encode(path(u).read_bytes()).decode()
    banned = {json.loads(l)["uid"] for f in a.exclude for l in open(f, encoding="utf-8")}
    rng = random.Random(4)
    guide, picks, used = [], [], set()
    for lab, hint in HINTS.items():
        cand = [u for u, c in ch.items() if c and c[0] == lab and u in F and path(u).exists() and u not in banned]
        # clearest first: fusion sure and Gemini agrees; rarer classes fall back to Gemini-agreed, then most probable
        tiers = [[u for u in cand if F[u][lab] > 0.95 and lab in gem.get(u, set())],
                 [u for u in cand if lab in gem.get(u, set())], cand]
        g = []
        for t in tiers:
            for u in sorted(t, key=lambda u: -F[u][lab]):
                if len(g) < a.guide_per_class and u not in g: g.append(u)
        used |= set(g)
        if g: guide.append({"label": lab, "hint": hint, "clips": [{"src": b64(u), "dur": 6.0} for u in g]})
        unsure = [u for u in cand if a.lo <= F[u][lab] <= a.hi and u not in used]
        rng.shuffle(unsure); picks += [(u, lab) for u in unsure[: a.n_per_class]]
    rng.shuffle(picks)
    items = [{"id": f"p{i:04d}", "dur": 6.0, "src": b64(u)} for i, (u, _) in enumerate(picks)]
    key = [{"id": f"p{i:04d}", "uid": u, "stratum": lab, "fused_p": round(F[u][lab], 3)} for i, (u, lab) in enumerate(picks)]
    tpl = (Path(__file__).with_name("label_kit_template.html")).read_text()
    html = (tpl.replace("__KIT_ID__", a.kit_id).replace("__CLIPS__", json.dumps(items))
               .replace("__GROUPS__", json.dumps(LABEL_GROUPS)).replace("__GUIDE__", json.dumps(guide)))
    a.out.write_text(html)
    a.out.with_suffix(".key.jsonl").write_text("".join(json.dumps(k) + "\n" for k in key))
    from collections import Counter
    print(f"{len(items)} clips to label, guide for {len(guide)} classes -> {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")
    print("per class:", dict(Counter(lab for _, lab in picks)), "| guide:", [g["label"] for g in guide])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
