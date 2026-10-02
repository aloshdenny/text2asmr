#!/usr/bin/env python3
"""Give the people who labelled Ear Check kits before the site an account, with their kit labels credited.

Each gets a profile waiting for whoever signs up with their email (the claim moves it, labels and all, onto the new
account; see migrations/20261002120000_claim_on_signup.sql), with their kit labels in public.imported_labels so the
leaderboard and contribution graph count them. Their fuse_labels.py name (adi, Alita, ...) is kept as the profile's fusion_name, so anything they label on
the site joins their existing labeller rather than a new crowd one. Idempotent: rerunning updates, never duplicates.

The contributors file holds emails, so keep it out of git:
  [{"email": "...", "username": "adi", "display_name": "Adi", "fusion_name": "adi",
    "files": ["adi_150.jsonl"], "default_date": "2026-10-01"}]

  python import_contributors.py --people contributors.json --labels-dir ~/t2a_samples/label_kits \\
      --db "$SUPABASE_DB_URL"            # postgres connection string (direct or session pooler)
"""
from __future__ import annotations
import argparse, datetime as dt, json, subprocess, sys
from pathlib import Path


def q(v) -> str:
    """SQL literal."""
    if v is None: return "null"
    if isinstance(v, list): return "array[" + ",".join(q(x) for x in v) + "]::text[]"
    return "'" + str(v).replace("'", "''") + "'"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=Path, required=True)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--dry-run", action="store_true", help="print the SQL instead of running it")
    a = ap.parse_args()
    sql = ["begin;"]
    for p in json.loads(a.people.read_text()):
        sql.append(f"select private.import_contributor({q(p['email'])}, {q(p['username'])}, {q(p.get('display_name'))}, {q(p.get('fusion_name'))});")
        uid = f"(select id from public.profiles where fusion_name = {q(p['fusion_name'])})"
        n = 0
        for f in p["files"]:
            for line in open(a.labels_dir / f, encoding="utf-8"):
                r = json.loads(line)
                if not r.get("uid") or r.get("skipped"): continue
                at = (dt.datetime.fromtimestamp(r["at"] / 1000, dt.timezone.utc) if r.get("at")
                      else dt.datetime.fromisoformat(p["default_date"] + "T12:00:00+00:00"))
                sql.append("insert into public.imported_labels (user_id, source_uid, labels, other_text, kit, created_at) values "
                           f"({uid}, {q(r['uid'])}, {q(r['labels'])}, {q(r.get('other') or None)}, {q(r.get('kit') or Path(f).stem)}, {q(at.isoformat())}) "
                           "on conflict (user_id, kit, source_uid) do update set labels = excluded.labels, other_text = excluded.other_text;")
                n += 1
        print(f"{p['username']:10} {n} kit labels", file=sys.stderr)
    sql.append("commit;")
    sql.append("select p.username, p.fusion_name, count(i.id) from public.profiles p left join public.imported_labels i on i.user_id = p.id "
               "where p.imported group by 1, 2 order by 3 desc;")
    text = "\n".join(sql)
    if a.dry_run:
        print(text)
        return 0
    r = subprocess.run(["psql", a.db, "-v", "ON_ERROR_STOP=1", "-q", "-A", "-F", " | "], input=text, text=True, capture_output=True)
    print(r.stdout.strip() or r.stderr.strip())
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
