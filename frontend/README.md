# ASMR Board (site)

The community labelling site for text2asmr: sign in with email and password, label 6-second clips with the keyboard,
climb the leaderboard, and build a GitHub-style contribution graph on your profile. Vite + React + supabase-js;
all data rules live in the database (`backend/supabase/migrations`), so the site is static.

```bash
cp .env.example .env.local      # SUPABASE_URL / SUPABASE_ANON_KEY of the local stack (backend/README.md)
npm install
npm run dev                     # http://localhost:5173
npm run build                   # typecheck + production build in dist/
```

Pages: `/` (stats, top listeners), `/label` (the labelling loop), `/leaderboard` (all time / 30 / 7 days),
`/u/<username>` (contribution graph, rank, streaks), `/guide` (what each label covers, with examples),
`/login` (email + password; kit labellers are told to set a password), `/auth/confirm` (email links),
`/set-password`, `/welcome` (username + 18+ confirmation), `/settings` (username, picture, delete account).

Keys on `/label`: `1`-`0`, `A`-`J` toggle labels · `Space` play/pause · `R` replay · `U` can't tell ·
`Enter` submit · `Esc` clear.

## No Supabase details in the bundle

The browser only talks to `<site>/sb/*`. In dev the Vite proxy (`vite.config.ts`) and on Vercel the edge function
`api/sb.ts` forward auth / REST / storage calls to Supabase and add the API key server-side; every other path is
refused. Email links point at `<site>/auth/confirm` (see `backend/supabase/templates`). `npm run build` and
`grep -r supabase.co dist/` finds nothing project-specific.

## Deploy (Vercel)

Live at https://asmrboard.vercel.app (Vercel project `asmrboard`, team clardentity; text2asmr.vercel.app redirects
there). Project env vars, server-side only (no `VITE_` prefix): `SUPABASE_URL`, `SUPABASE_ANON_KEY` (the
publishable / anon key). Deploy from this folder:

```bash
npx vercel deploy --prod --scope clardentity
```
