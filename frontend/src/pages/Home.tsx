import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import Avatar from '../components/Avatar'
import { AppLayout } from '../components/Layouts'
import Ring from '../components/Ring'
import { ButtonLink, Card, CardHeader, cx } from '../components/ui'
import { useAuth } from '../lib/auth'
import { useLeaderboard } from '../lib/stats'
import { supabase } from '../lib/supabase'

type Progress = { corpus_items: number; human_labelled_items: number; ai_labelled_items: number }

/** Whole percent; anything above zero but under one shows as 1%. */
const pct = (part: number, whole: number) => {
  const p = whole > 0 ? (100 * part) / whole : 0
  return p > 0 && p < 1 ? 1 : Math.round(p)
}

const STEPS = [
  ['Listen', 'One short clip at a time. Headphones help.'],
  ['Tick what you hear', 'Whispers, tapping, crinkles, brushing. Usually more than one.'],
  ['Stay independent', 'Every clip goes to several labellers, and nobody sees anyone else’s answer.'],
]

export default function Home() {
  const { session, profile } = useAuth()
  const [progress, setProgress] = useState<Progress | null>(null)
  const top = useLeaderboard(30, 5)
  useEffect(() => {
    supabase.rpc('dataset_progress').then(({ data }) => setProgress((data as Progress[] | null)?.[0] ?? null))
  }, [])
  const human = progress ? pct(progress.human_labelled_items, progress.corpus_items) : 0
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
          <Ring percent={human} color="var(--ring-green)" label="data labelled by people" caption="on ASMR Board and the Ear Check kits" />
          <Ring percent={progress ? 100 - human : 0} color="var(--brand)" label="data labelled by AI" caption="our CLAP-ASMR model" />
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
