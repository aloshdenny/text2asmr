# ASMR Board backend (Supabase)

Community labelling for the ASMR-CLAP ontology. Listeners label 6-second clips on the site in `frontend/`; their
labels go back into `scripts/fuse_labels.py`, where each listener is one more labeller whose reliability is learned
per class.

```
supabase/migrations/   schema, row-level security, labelling + leaderboard + admin RPCs, storage buckets
supabase/tests/        pgTAP tests (supabase test db)
sync/push_clips.py     queue clips: unsure fused clips, unlabelled (mined) windows, hidden controls
sync/push_guide.py     publish the sound guide (example clips per label, public)
sync/pull_labels.py    export labels as judges/human/crowd.jsonl for fuse_labels.py
sync/import_contributors.py  accounts for people who labelled Ear Check kits before the site, kit labels credited
supabase/templates/    auth emails (confirm sign-up, set password) linking to <site>/auth/confirm
```

## How it stays blind and multi-judge

- A listener only ever receives an opaque clip id and a 15-minute signed URL for the **one clip assigned to them**
  (`next_clip()`); storage refuses every other object. Clip sources, fused probabilities, vote counts and other
  people's answers are never readable by listeners.
- Every clip needs `target_votes` answers (default 3). Clips already in progress are served first so they reach
  their quorum; nobody gets the same clip twice; open assignments expire after 20 minutes.
- About 10% of the queue are **controls** (clips people already labelled), indistinguishable from the rest. The
  fusion uses them to learn how reliable each listener is (`crowd:<username>`, weak prior `--crowd-prior 3`).
- An answer is accepted only after at least 1.5 s of playback and 2 s since assignment.
- 18+ gate at sign-up: clips include intimate vocal sounds.

## Local development

Needs Docker (Colima works: `brew install colima docker && colima start`).

```bash
cd backend
supabase start -x realtime,edge-runtime,logflare,vector,imgproxy,supavisor
supabase test db                      # 30 pgTAP tests
```

Sign-in emails land in Mailpit at http://127.0.0.1:54324. Queue some clips:

```bash
export SUPABASE_URL=http://127.0.0.1:54321
export SUPABASE_SERVICE_ROLE_KEY=$(supabase status -o json | python3 -c "import json,sys; print(json.load(sys.stdin)['SERVICE_ROLE_KEY'])")
python sync/push_clips.py --manifest manifest.jsonl --batch local-test   # rows: {"uid", "path", "kind"}
```

If a Docker Desktop leftover breaks image pulls (`docker-credential-desktop` not found), point the CLI at a clean
config: `DOCKER_CONFIG=$(mktemp -d) DOCKER_HOST=unix://$HOME/.colima/default/docker.sock supabase start ...`.

## Accounts

Email + password. People who labelled Ear Check kits before the site get an account made for them
(`import_contributors.py`, run with the database URL; emails live in a local file, never in git): no password,
their kit labels in `imported_labels` (they count on the leaderboard and contribution graph), and their
fuse_labels.py name kept as `fusion_name`. When they sign in or up, `account_status()` reports `needs_password`
for exactly those accounts and the site emails them a link to set one. Deleting an account removes the sign-in and
takes the profile off every public list; the labels stay (under a tombstoned profile).

## Hosted project

Project `t2a` (ref gdjfvjsxsmwjzqpgbapk). Schema: `supabase db push --db-url "$SUPABASE_DB_URL"` (session
pooler URL). Auth settings that live in the dashboard (or the Management API), mirroring `config.toml`:
site URL `https://asmrboard.vercel.app`, redirect URL `https://asmrboard.vercel.app/**`, minimum password length
8, email confirmation on, the two templates in `supabase/templates/`, and custom SMTP (the built-in sender only
mails the project's team members, so set-password emails to other people need it).
Feed it from the research server (where the pool clips live):

```bash
python sync/push_clips.py --fused fused.jsonl --clips D:/t2a/pool D:/t2a/pool_yt D:/t2a/pool_spray \
    --humans judges/human/*.jsonl --n-unsure 2000 --batch 2026-10-a
python sync/push_guide.py --plan kit5_plan.json --clips D:/t2a/pool D:/t2a/pool_yt D:/t2a/pool_spray
python sync/pull_labels.py --out judges/human/crowd.jsonl
python scripts/fuse_labels.py ... --humans judges/human/*.jsonl      # crowd.jsonl included
```

The service-role key bypasses every check: keep it on the server, never in the frontend or in git.
