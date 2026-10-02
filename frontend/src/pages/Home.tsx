import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import Avatar from '../components/Avatar'
import { AppLayout } from '../components/Layouts'
import { ButtonLink, Card, CardHeader, Stat } from '../components/ui'
import { useAuth } from '../lib/auth'
import { useLeaderboard } from '../lib/stats'
import { supabase } from '../lib/supabase'

type Stats = { labels: number; labellers: number; clips: number; clips_complete: number }

const STEPS = [
  ['Listen', 'One six-second clip at a time. Headphones help.'],
  ['Tick what you hear', 'Whispers, tapping, crinkles, brushing — usually more than one.'],
  ['Stay independent', 'Every clip goes to several listeners, and nobody sees anyone else’s answer.'],
]

export default function Home() {
  const { session, profile } = useAuth()
  const [stats, setStats] = useState<Stats | null>(null)
  const top = useLeaderboard(30, 5)
  useEffect(() => {
    supabase.rpc('site_stats').then(({ data }) => setStats((data as Stats[] | null)?.[0] ?? null))
  }, [])
  const n = (v?: number) => (v ?? 0).toLocaleString()
  const signedIn = Boolean(session && profile?.onboarded)
  return (
    <AppLayout>
      <section className="max-w-2xl pb-4 pt-6 sm:pt-12">
        <p className="text-sm font-medium text-ink-muted">A community dataset for text2asmr</p>
        <h1 className="mt-3 text-4xl font-semibold tracking-tight text-ink sm:text-5xl">Teach machines to hear ASMR.</h1>
        <p className="mt-4 text-[17px] leading-relaxed text-ink-secondary">
          Listen to short clips and tick the sounds you hear. Your labels train the open sound classifier behind
          text2asmr, and every one of them counts toward your place on the board.
        </p>
        <div className="mt-7 flex flex-wrap gap-2.5">
          {signedIn ? (
            <ButtonLink to="/label" variant="primary" size="lg">Start labelling</ButtonLink>
          ) : (
            <ButtonLink to="/signup" variant="primary" size="lg">Create an account</ButtonLink>
          )}
          <ButtonLink to="/guide" size="lg">Hear the sound guide</ButtonLink>
        </div>
        <p className="mt-4 text-xs text-ink-muted">18+ only — some clips contain intimate vocal sounds.</p>
      </section>

      <section className="mt-10 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat value={n(stats?.labels)} label="labels given" />
        <Stat value={n(stats?.labellers)} label="listeners" />
        <Stat value={n(stats?.clips_complete)} label="clips fully labelled" />
        <Stat value={n(stats?.clips)} label="clips in the queue" />
      </section>

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
          <CardHeader title="Top listeners" description="Last 30 days" action={<Link to="/leaderboard" className="text-sm text-brand hover:underline">All</Link>} />
          {top && top.length === 0 && <p className="text-sm text-ink-muted">No labels yet — be the first.</p>}
          <ol className="divide-y divide-hairline">
            {(top ?? []).map((r) => (
              <li key={r.username} className="flex items-center gap-3 py-2.5 first:pt-0 last:pb-0">
                <span className="w-4 text-right font-mono text-xs text-ink-muted">{r.rank}</span>
                <Avatar name={r.username} url={r.avatar_url} size={24} />
                <Link to={`/u/${r.username}`} className="min-w-0 flex-1 truncate text-sm text-ink hover:underline">{r.display_name || r.username}</Link>
                <span className="font-mono text-sm tabular-nums text-ink-secondary">{r.labelled.toLocaleString()}</span>
              </li>
            ))}
          </ol>
        </Card>
      </section>
    </AppLayout>
  )
}
