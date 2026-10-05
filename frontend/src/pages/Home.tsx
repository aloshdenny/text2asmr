import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import Avatar from '../components/Avatar'
import { AppLayout } from '../components/Layouts'
import Ring from '../components/Ring'
import { ButtonLink, Card, CardHeader, cx } from '../components/ui'
import { useAuth } from '../lib/auth'
import { useLeaderboard } from '../lib/stats'
import { supabase } from '../lib/supabase'

type Progress = { foundation_items: number; foundation_human_labelled: number; corpus_recordings: number; ai_labelled_recordings: number }

/** Percent to one decimal place; anything above zero but under 0.1 shows as 0.1%. */
const pct = (part: number, whole: number) => {
  const p = whole > 0 ? (100 * part) / whole : 0
  return p > 0 && p < 0.1 ? 0.1 : Math.round(p * 10) / 10
}

const PROGRESS_KEY = 'asmrboard-progress'

/** The last numbers this browser saw, so the rings draw on the first frame (storage can be unavailable). */
function cachedProgress(): Progress | null {
  try {
    return JSON.parse(localStorage.getItem(PROGRESS_KEY) ?? 'null') as Progress | null
  } catch {
    return null
  }
}

/** The ring numbers: the CDN-cached /api/progress (a once-a-day snapshot), or the RPC where that route is absent (dev). */
async function fetchProgress(): Promise<Progress | null> {
  try {
    const r = await fetch('/api/progress')
    if (r.ok && r.headers.get('content-type')?.includes('json')) return (await r.json()) as Progress
  } catch {
    /* fall through to the RPC */
  }
  const { data } = await supabase.rpc('dataset_progress')
  return (data as Progress[] | null)?.[0] ?? null
}

const STEPS = [
  ['Listen', 'One short clip at a time. Headphones help.'],
  ['Tick what you hear', 'Whispers, tapping, crinkles, brushing. Usually more than one.'],
  ['Stay independent', 'Every clip goes to several labellers, and nobody sees anyone else’s answer.'],
]

export default function Home() {
  const { session, profile } = useAuth()
  const [progress, setProgress] = useState<Progress | null>(cachedProgress)
  const top = useLeaderboard(30, 5)
  useEffect(() => {
    fetchProgress().then((p) => {
      if (!p) return
      setProgress(p)
      try {
        localStorage.setItem(PROGRESS_KEY, JSON.stringify(p))
      } catch {
        /* private mode: just no head start next time */
      }
    })
  }, [])
  const n = (v?: number) => (v ?? 0).toLocaleString()
  const human = progress ? pct(progress.foundation_human_labelled, progress.foundation_items) : 0
  const ai = progress ? pct(progress.ai_labelled_recordings, progress.corpus_recordings) : 0
  const signedIn = Boolean(session && profile?.onboarded)
  return (
    <AppLayout>
      <section className="max-w-2xl pb-4 pt-6 sm:pt-12">
        <p className="text-sm font-medium text-ink-muted">A community dataset for text2asmr</p>
        <h1 className="mt-3 text-4xl font-semibold tracking-tight text-ink sm:text-5xl">Teach machines to generate ASMR.</h1>
        <p className="mt-4 text-[17px] leading-relaxed text-ink-secondary">
          We’re an open platform where anyone can help label audio for the OpenASMR project. Every clip you label
          counts toward your place on the board.
        </p>
        <div className="mt-7 flex flex-wrap gap-2.5">
          {signedIn ? (
            <>
              <ButtonLink to="/label" variant="primary" size="lg">Start labelling</ButtonLink>
            </>
          ) : (
            <ButtonLink to="/signup" variant="primary" size="lg">Create an account</ButtonLink>
          )}
        </div>
        <p className="mt-4 text-xs text-ink-muted">18+ only. Some clips contain intimate vocal sounds.</p>
      </section>

      <Card className="mt-10 py-8">
        <div className="flex flex-wrap items-start justify-center gap-x-6 gap-y-8 sm:gap-x-24">
          <Ring
            percent={human}
            color="var(--ring-green)"
            label="labelled by people"
            caption={progress ? `${n(progress.foundation_human_labelled)} of ${n(progress.foundation_items)} core training clips` : ''}
          />
          <Ring
            percent={ai}
            color="var(--brand)"
            label="labelled by AI (CLAP-ASMR)"
            caption={progress ? `${n(progress.ai_labelled_recordings)} of ${n(progress.corpus_recordings)} recordings` : ''}
          />
        </div>
      </Card>

      <section className="mt-3 grid gap-3 md:grid-cols-[1.4fr_1fr]">
        <Card>
          <CardHeader title="How it works" />
          <ol className="space-y-4">
            {STEPS.map(([t, d], i) => (
              <li key={t} className="flex gap-3">
                <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-hairline-strong font-mono text-[11px] text-ink-secondary">{i + 1}</span>
                <div>
                  <p className="text-sm font-medium text-ink">{t}</p>
                  <p className="text-sm text-ink-muted">{d}</p>
                </div>
              </li>
            ))}
          </ol>
        </Card>
        <Card>
          <CardHeader title="Top labellers" description="Last 30 days" action={<Link to="/leaderboard" className="text-sm text-brand hover:underline">All</Link>} />
          {top && top.length === 0 && <p className="text-sm text-ink-muted">No labels yet. Be the first.</p>}
          <ol className="divide-y divide-hairline">
            {(top ?? []).map((r) => {
              const me = profile?.username === r.username
              return (
                <li key={r.username} className="flex items-center gap-3 py-2.5 first:pt-0 last:pb-0">
                  <span className="w-4 text-right font-mono text-xs text-ink-muted">{r.rank}</span>
                  <Avatar name={r.username} url={r.avatar_url} size={24} />
                  <Link to={`/u/${r.username}`} className={cx('min-w-0 flex-1 truncate text-sm text-ink hover:underline', me && 'font-medium')}>
                    {r.username}
                    {me && <span className="ml-1 text-ink-muted">(You)</span>}
                  </Link>
                  <span className="font-mono text-sm tabular-nums text-ink-secondary">{r.labelled.toLocaleString()}</span>
                </li>
              )
            })}
          </ol>
        </Card>
      </section>
    </AppLayout>
  )
}
