# Ear Check (site)

The community labelling site for text2asmr: sign in with an email link, label 6-second clips with the keyboard,
climb the leaderboard, and build a GitHub-style contribution graph on your profile. Vite + React + supabase-js;
all data rules live in the database (`backend/supabase/migrations`), so the site is static.

```bash
cp .env.example .env.local      # local stack values: see backend/README.md
npm install
npm run dev                     # http://localhost:5173
npm run build                   # typecheck + production build in dist/
```

Pages: `/` (stats, top listeners), `/label` (the labelling loop), `/leaderboard` (all time / 30 / 7 days),
`/u/<username>` (contribution graph, rank, streaks), `/guide` (what each label covers, with examples),
`/login`, `/welcome` (username + 18+ confirmation).

Keys on `/label`: `1`-`0`, `A`-`J` toggle labels · `Space` play/pause · `R` replay · `U` can't tell ·
`Enter` submit · `Esc` clear.

## Deploy (Vercel)

Import the GitHub repo, set **Root Directory** to `frontend`, framework preset Vite, and set
`VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY` (the anon / publishable key, safe in the browser) and optionally
`VITE_GITHUB_AUTH=1`. `vercel.json` rewrites every path to the app so deep links like `/u/name` work.
