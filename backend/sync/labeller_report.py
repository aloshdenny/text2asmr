#!/usr/bin/env python3
"""Who is listening: the site's weak-labeller review list (admin_labeller_quality), weakest first.

follow is the core number: 1 = a person's answers track what is in each clip, 0 = they would fit any clip just as well
(a random clicker, or someone who ticks the same thing every time). Statuses:
  weak       answers do not follow the audio, or two warning signs: left out of consensus, the people ring and exports
  watch      one warning sign: served more control clips and hidden repeats until it settles
  too early  fewer than 15 answers with a reference
  ok         answers follow the audio, no warning signs
Use it before paying out anything tied to a milestone: only "ok" labellers have shown they listen.

Below the list come the groups of accounts seen on one device, IP or network (scrambled marks the /sb proxy adds;
nothing identifies a person). One device behind several accounts is a warning sign in the rating; a shared IP or
network is only shown, since households, offices and mobile networks share those.

  python labeller_report.py --env-file ~/.t2a/asmrboard.env [--all] [--json]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from push_clips import Site, load_env_file


def fmt(v, nd=2):
    return "" if v is None else f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include people with fewer than 5 answers")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--env-file", type=Path, default=None)
    a = ap.parse_args()
    load_env_file(a.env_file)
    site = Site()
    rows = site.rpc("admin_labeller_quality")
    groups = site.rpc("admin_shared_marks")
    if not a.all: rows = [r for r in rows if r["answers"] >= 5 or r["status"] in ("weak", "watch")]
    if a.json:
        print(json.dumps({"labellers": rows, "shared": groups}, indent=1, default=str)); return 0
    cols = [("username", 16), ("status", 9), ("answers", 7), ("refs", 5), ("follow", 6), ("±", 5), ("repeats", 7),
            ("answer s", 8), ("rush", 5), ("top label", 16), ("devices", 7), ("browser", 15), ("flags", 0)]
    print("  ".join(c.ljust(w) for c, w in cols))
    for r in rows:
        top = f"{r['top_label'] or ''} {fmt(r['top_label_share'])}".strip()
        vals = [r["username"] + (" (guest)" if r["guest"] else ""), r["status"], r["answers"], r["refs"], fmt(r["follow"]),
                fmt(r["follow_se"]), f"{r['retests']}" + (f"/{fmt(r['retest_agreement'])}" if r["retests"] else ""),
                fmt(r["median_answer_s"], 1), fmt(r["rush_rate"]), top,
                f"{r['devices']}" + (f" +{r['device_peers']}" if r["device_peers"] else ""), (r["browsers"] or [""])[0],
                ", ".join(r["flags"])]
        print("  ".join(str(v).ljust(w) for v, (_, w) in zip(vals, cols)))
    counts = {s: sum(r["status"] == s for r in rows) for s in ("ok", "watch", "weak", "too early")}
    print(f"\n{len(rows)} labellers: " + ", ".join(f"{n} {s}" for s, n in counts.items()))
    print(f"\nShared devices / IPs / networks ({len(groups)} groups; +N under devices = other accounts on the same device):")
    for g in groups:
        print(f"  {g['kind']:<6} {g['mark']}  {g['accounts']} accounts, {g['labels']} labels, last seen {str(g['last_seen'])[:16]}:  "
              + ", ".join(g["members"]))
    if not groups: print("  none yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
