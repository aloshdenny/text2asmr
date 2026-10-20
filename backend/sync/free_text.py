#!/usr/bin/env python3
"""What labellers type under "something else", boxed into menu labels.

pull_labels.py adds the label a free-text answer names ("Giggling" -> laughing) to that answer, and to the labels it
was offered, so an answer given before the label existed still counts. Text that maps to nothing is what the next
label could be: run this file to see it grouped, and give a sound its own label once several people keep typing it.

  python free_text.py --env-file ~/.t2a/asmrboard.env
"""
from __future__ import annotations
import argparse, re, sys
from collections import Counter, defaultdict
from pathlib import Path

#: free text -> menu label (checked 2026-10-20 against all 37 free-text answers)
MAP = {
    "laughing": r"laugh|giggl|chuckl",
    "gasping": r"gasp",
    "spanking": r"spank|beating",
    "clapping": r"\bclap",
    "scratching": r"writing|drag",
    "sticky": r"velcro|hook.and.loop|slime",
    "liquid": r"\brain",
    "oral sounds": r"slurp|suck",
    "crying": r"\bcr(y|ies|ying)\b|sob",           # not on the menu yet: kept here so the report counts it
}
_MAP = {k: re.compile(p, re.I) for k, p in MAP.items()}


def labels_in(text: str | None, menu: set[str] | None = None) -> list[str]:
    """Menu labels the text names (only ones on `menu`, when given)."""
    if not text: return []
    return [k for k, p in _MAP.items() if p.search(text) and (menu is None or k in menu)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", type=Path, default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from push_clips import Site, load_env_file
    load_env_file(a.env_file)
    rows = [r for r in Site().rpc("admin_export_labels") if (r.get("other_text") or "").strip()]
    boxed, people, loose = Counter(), defaultdict(set), []
    for r in rows:
        hit = labels_in(r["other_text"])
        for k in hit: boxed[k] += 1; people[k].add(r["labeller"])
        if not hit: loose.append(r["other_text"].strip())
    print(f"{len(rows)} free-text answers")
    for k, n in boxed.most_common(): print(f"  {n:3d}  {k}  ({len(people[k])} people)")
    print(f"not boxed ({len(loose)}):")
    for t, n in Counter(x.lower() for x in loose).most_common(): print(f"  {n:3d}  {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
