import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import Avatar from '../components/Avatar'
import { AppLayout } from '../components/Layouts'
import Ring from '../components/Ring'
import PixelCritter from '../components/PixelCritter'
import { Mascot } from '../components/TopBar'
import { ButtonLink, Card, CardHeader, cx } from '../components/ui'
import { useAuth } from '../lib/auth'
import { useLeaderboard } from '../lib/stats'
import { supabase } from '../lib/supabase'

type Progress = {
  foundation_items: number; foundation_human_labelled: number; corpus_recordings: number; ai_labelled_recordings: number
  human_labels?: number // every answer people gave (absent in numbers cached before 2026-10-17)
}

/** Percent to one decimal place; anything above zero but under 0.1 shows as 0.1%. */
const pct = (part: number, whole: number) => {
  const p = whole > 0 ? (100 * part) / whole : 0
  return p > 0 && p < 0.1 ? 0.1 : Math.round(p * 10) / 10
}

/** The people ring counts labels -- every answer people have given -- toward the next milestone. Counting clips with at
 *  least one label stood still whenever people were adding judges to clips already started (742 for a week of new
 *  labels); every answer moves this one. The caption says it is a milestone. */
const MILESTONES = [250, 500, 1000, 2000, 5000, 10000]
const milestone = (done: number) => MILESTONES.find((m) => m > done) ?? (Math.floor(done / 10000) + 1) * 10000

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
  const answers = progress ? progress.human_labels ?? progress.foundation_human_labelled : 0
  const goal = milestone(answers)
  // never round up to a full ring before the milestone is actually reached (4,999 of 5,000 is 99.9%, not 100%)
  const human = progress ? Math.min(pct(answers, goal), answers < goal ? 99.9 : 100) : 0
  const signedIn = Boolean(session && profile?.onboarded)
  return (
    <AppLayout>
      <section className="grid items-center gap-4 pb-6 pt-2 sm:pt-10 md:grid-cols-[1.05fr_1fr] md:gap-10">
        <Choir />
        <div className="text-center md:text-left">
          <h1 className="text-balance text-4xl font-black tracking-tight text-ink sm:text-5xl">A music project to teach machines to generate ASMR.</h1>
          <p className="mx-auto mt-4 max-w-md text-lg font-semibold leading-relaxed text-ink-secondary md:mx-0">
            Listen to short clips and tick the sounds you hear. Every clip you label counts toward your place on the board.
          </p>
          <div className="mx-auto mt-7 flex max-w-xs flex-col gap-3 md:mx-0">
            <ButtonLink to={signedIn ? '/label' : '/start'} variant="primary" size="lg" className="w-full">{signedIn ? 'Keep labelling' : 'Get started'}</ButtonLink>
            {!signedIn && <ButtonLink to="/login" variant="secondary" size="lg" className="w-full">I already have an account</ButtonLink>}
          </div>
          <p className="mt-4 text-xs font-semibold text-ink-muted">{signedIn ? '' : 'No sign-up needed. '}18+ only. Some clips contain intimate vocal sounds.</p>
        </div>
      </section>

      <Card className="mt-8 py-10">
        <div className="flex justify-center">
          <Ring
            size={204}
            percent={human}
            color="var(--ring-green)"
            label="labelled by people"
            caption={progress ? `${n(answers)} of ${n(goal)} labels, the next milestone` : ''}
          />
        </div>
      </Card>

      <section className="mt-3 grid gap-3 sm:grid-cols-3">
        {STEPS.map(([t, d], i) => (
          <Card key={t} className="flex flex-col items-center text-center">
            <span className="mb-3 overflow-hidden rounded-2xl"><PixelCritter seed={`step-${i}`} kind={STEP_CRITTERS[i]} size={64} /></span>
            <p className="text-base font-extrabold text-ink">{t}</p>
            <p className="mt-1 text-sm font-semibold text-ink-muted">{d}</p>
          </Card>
        ))}
      </section>

      <Card className="mt-3">
        <CardHeader title="Top labellers" description="Last 30 days" action={<Link to="/leaderboard" className="text-sm font-extrabold uppercase tracking-[0.06em] text-brand hover:underline">All</Link>} />
        {top && top.length === 0 && <p className="text-sm font-semibold text-ink-muted">No labels yet. Be the first.</p>}
        <ol className="divide-y-2 divide-hairline">
          {(top ?? []).map((r) => {
            const me = profile?.username === r.username
            return (
              <li key={r.username} className="flex items-center gap-3 py-3 first:pt-0 last:pb-0">
                <span className="w-5 text-right text-sm font-black text-ink-muted">{r.rank}</span>
                <Avatar name={r.username} url={r.avatar_url} size={40} />
                <Link to={`/u/${r.username}`} className={cx('min-w-0 flex-1 truncate text-[15px] font-bold text-ink hover:underline', me && 'text-brand')}>
                  {r.username}
                  {me && <span className="ml-1 text-ink-muted">(You)</span>}
                </Link>
                <span className="text-[15px] font-extrabold tabular-nums text-ink-secondary">{r.labelled.toLocaleString()}</span>
              </li>
            )
          })}
        </ol>
      </Card>
    </AppLayout>
  )
}

const STEP_CRITTERS = ['bunny', 'cat', 'owl'] as const

/** The hero: a little choir of critters with headphones, bobbing out of step while sound ripples out of the middle. */
function Choir() {
  const seats: { seed: string; kind: 'cat' | 'bunny' | 'bear' | 'fox' | 'frog' | 'mouse' | 'owl'; x: number; y: number; s: number; d: number }[] = [
    { seed: 'choir-a', kind: 'fox', x: 2, y: 44, s: 104, d: 0.3 },
    { seed: 'choir-b', kind: 'bear', x: 74, y: 40, s: 112, d: 1.1 },
    { seed: 'choir-c', kind: 'frog', x: 18, y: 2, s: 92, d: 0.7 },
    { seed: 'choir-d', kind: 'mouse', x: 62, y: 0, s: 96, d: 1.6 },
  ]
  return (
    <div className="relative mx-auto aspect-[1.5] w-full max-w-[460px]" aria-hidden="true">
      {[0, 0.7, 1.4].map((delay) => (
        <span key={delay} className="wave-out absolute left-1/2 top-[47%] h-[62%] w-[42%] -translate-x-1/2 -translate-y-1/2 rounded-full border-[3px] border-brand-border" style={{ animationDelay: `${delay}s` }} />
      ))}
      {seats.map((c) => (
        <span key={c.seed} className="critter-bob absolute overflow-hidden rounded-3xl border-2 border-hairline" style={{ left: `${c.x}%`, top: `${c.y}%`, width: `${c.s / 4.4}%`, animationDelay: `${c.d}s` }}>
          <PixelCritter seed={c.seed} kind={c.kind} size={256} className="h-auto w-full" />
        </span>
      ))}
      <span className="critter-bob absolute left-1/2 top-[20%] w-[36%] -translate-x-1/2 overflow-hidden rounded-[2rem] border-2 border-brand-border" style={{ animationDelay: '0s' }}>
        <Mascot size={256} className="h-auto w-full" />
      </span>
    </div>
  )
}
