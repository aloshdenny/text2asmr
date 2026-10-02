#!/usr/bin/env python3
"""Export the crowd's labels in the people-label format scripts/fuse_labels.py reads (--humans).

Each listener becomes one labeller, "crowd:<username>", whose reliability per class the fusion learns from the
control clips and from agreement with everyone else. "Can't tell" with no labels is an abstention (skipped);
"can't tell" with a best guess is kept as a vote with the guess.

  export SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=...
  python pull_labels.py --out judges/human/crowd.jsonl [--min-labels 20]
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from push_clips import Site


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-labels", type=int, default=20, help="leave out listeners with fewer labels (too few to learn a reliability)")
    a = ap.parse_args()
    site = Site()
    rows = site.rpc("admin_export_labels")
    menu = [o["key"] for o in sorted(site_menu(site), key=lambda o: o["sort"])]
    n = Counter(r["username"] for r in rows)
    keep = [r for r in rows if n[r["username"]] >= a.min_labels]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in keep:
            fh.write(json.dumps({"who": f"crowd:{r['username']}", "uid": r["source_uid"], "labels": r["labels"],
                                 "skipped": r["unsure"] and not r["labels"], "unsure": r["unsure"],
                                 "other": r["other_text"], "menu": menu, "kind": r["kind"],
                                 "listen_ms": r["listen_ms"], "at": r["created_at"]}) + "\n")
    print(f"{len(rows)} labels from {len(n)} listeners; wrote {len(keep)} from {sum(v >= a.min_labels for v in n.values())} "
          f"listeners with >= {a.min_labels} labels -> {a.out}")
    return 0


def site_menu(site: Site) -> list[dict]:
    import requests
    r = requests.get(f"{site.url}/rest/v1/label_options?select=key,sort", headers=site.h, timeout=30)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    raise SystemExit(main())
