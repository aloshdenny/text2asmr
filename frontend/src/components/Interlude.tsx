import { useEffect, useEffectEvent, useRef, useState, type RefObject } from 'react'
import { supabase } from '../lib/supabase'
import { Button, Card, Kbd, cx } from './ui'

/** A short break between labels (next_interlude): "Real or AI?" on one clip, or "Which is hotter?" on two. The
 *  server picks it, keeps the answer until the guess is in, and serves only these clips. */
export type InterludeKind = 'rai' | 'hot'

type RaiResult = { correct: boolean; was_ai: boolean; fooled_pct: number | null; others: number; my_correct: number; my_total: number }
type HotResult = { agree_pct: number | null; judges: number }

/** Seconds actually played: the server wants a real listen before it accepts an answer. */
function listenedMs(a: HTMLAudioElement | null): number {
  if (!a) return 0
  let t = 0
  for (let i = 0; i < a.played.length; i++) t += a.played.end(i) - a.played.start(i)
  return Math.round(t * 1000)
}

function Player({ src, label, audioRef, onPlay, onProgress, autoPlay }: {
  src: string
  label: string
  audioRef: RefObject<HTMLAudioElement | null>
  onPlay?: () => void
  onProgress?: () => void
  autoPlay?: boolean
}) {
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState({ at: 0, dur: 0 })
  useEffect(() => {
    if (autoPlay) audioRef.current?.play().catch(() => {})
  }, [autoPlay, audioRef])
  const toggle = () => {
    const a = audioRef.current
    if (!a) return
    if (a.paused) a.play().catch(() => {})
    else a.pause()
  }
  const pct = time.dur ? Math.min(100, (time.at / time.dur) * 100) : 0
  return (
    <div className="flex items-center gap-3">
      <button
        onClick={toggle}
        aria-label={playing ? `Pause ${label}` : `Play ${label}`}
        className={cx('flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-brand text-white transition-colors hover:bg-brand-dark', playing && 'play-ring')}
      >
        {playing ? (
          <svg viewBox="0 0 24 24" className="h-5 w-5 fill-current" aria-hidden><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>
        ) : (
          <svg viewBox="0 0 24 24" className="ml-0.5 h-5 w-5 fill-current" aria-hidden><path d="M8 5.5v13l11-6.5z" /></svg>
        )}
      </button>
      <div className="min-w-0 flex-1">
        <div className="mb-1.5 flex items-baseline justify-between text-xs">
          <span className="font-medium text-ink">{label}</span>
          <span className="font-mono tabular-nums text-ink-muted">{time.at.toFixed(1)} / {time.dur.toFixed(1)}s</span>
        </div>
        <div className="h-1.5 overflow-hidden rounded-full bg-surface-hover">
          <div className="h-full rounded-full bg-brand" style={{ width: `${pct}%` }} />
        </div>
      </div>
      <audio
        ref={audioRef}
        src={src}
        preload="auto"
        onPlay={() => {
          setPlaying(true)
          onPlay?.()
        }}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onTimeUpdate={(e) => {
          setTime({ at: e.currentTarget.currentTime, dur: e.currentTarget.duration || 0 })
          onProgress?.()
        }}
        onLoadedMetadata={(e) => setTime({ at: 0, dur: e.currentTarget.duration || 0 })}
      />
    </div>
  )
}

export default function Interlude({ kind, srcA, srcB, autoPlay, onDone, onError }: {
  kind: InterludeKind
  srcA: string
  srcB: string | null
  autoPlay: boolean
  onDone: () => void
  onError: (message: string) => void
}) {
  const a = useRef<HTMLAudioElement>(null)
  const b = useRef<HTMLAudioElement>(null)
  const [heard, setHeard] = useState({ a: 0, b: 0 }) // ms played of each clip: the answer buttons unlock with it
  const [busy, setBusy] = useState(false)
  const [rai, setRai] = useState<RaiResult | null>(null)
  const [hot, setHot] = useState<HotResult | null>(null)
  const done = rai !== null || hot !== null

  const ready = kind === 'rai' ? heard.a >= 1500 : heard.a >= 1000 && heard.b >= 1000

  const guess = async (choice: 1 | 2) => {
    if (busy || done) return
    if (!ready) return onError(kind === 'rai' ? 'Listen to the clip first.' : 'Listen to both clips first.')
    a.current?.pause()
    b.current?.pause()
    setBusy(true)
    if (kind === 'rai') {
      const { data, error } = await supabase.rpc('answer_rai', { p_guess_ai: choice === 2, p_listen_ms: heard.a })
      setBusy(false)
      if (error) return onError(error.message)
      setRai((data as RaiResult[])[0])
    } else {
      const { data, error } = await supabase.rpc('answer_hot', {
        p_pick: choice === 1 ? 'a' : 'b',
        p_listen_a_ms: heard.a,
        p_listen_b_ms: heard.b,
      })
      setBusy(false)
      if (error) return onError(error.message)
      setHot((data as HotResult[])[0])
    }
  }

  const skip = async () => {
    if (busy) return
    setBusy(true)
    await supabase.rpc('skip_interlude')
    setBusy(false)
    onDone()
  }

  const onKey = useEffectEvent((e: KeyboardEvent) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return
    if (done) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        onDone()
      }
      return
    }
    if (e.key === '1' || e.key === '2') {
      e.preventDefault()
      guess(e.key === '1' ? 1 : 2)
    } else if (e.key === ' ' && kind === 'rai') {
      e.preventDefault()
      const el = a.current
      if (el?.paused) el.play().catch(() => {})
      else el?.pause()
    }
  })
  useEffect(() => {
    const f = (e: KeyboardEvent) => onKey(e)
    window.addEventListener('keydown', f)
    return () => window.removeEventListener('keydown', f)
  }, [])

  const update = () => setHeard({ a: listenedMs(a.current), b: listenedMs(b.current) })

  return (
    <Card className="hp-rise">
      <p className="text-[11px] font-semibold uppercase tracking-wider text-ink-muted">Quick break</p>
      <h2 className="mt-1 text-lg font-semibold text-ink">{kind === 'rai' ? 'Real or AI?' : 'Which is hotter?'}</h2>
      <p className="mt-0.5 text-sm text-ink-muted">
        {kind === 'rai' ? 'Is this a real person, or our AI copying their voice?' : 'Listen to both, then pick the one you like more.'}
      </p>

      <div className="mt-5 space-y-4">
        <Player src={srcA} label={kind === 'rai' ? 'The clip' : 'A'} audioRef={a} autoPlay={autoPlay} onPlay={() => b.current?.pause()} onProgress={update} />
        {kind === 'hot' && srcB && <Player src={srcB} label="B" audioRef={b} onPlay={() => a.current?.pause()} onProgress={update} />}
      </div>

      {!done ? (
        <>
          <div className="mt-5 grid grid-cols-2 gap-2">
            <Button size="lg" variant="secondary" onClick={() => guess(1)} disabled={busy || !ready}>
              {kind === 'rai' ? 'Real' : 'A is hotter'} <Kbd className="max-sm:hidden">1</Kbd>
            </Button>
            <Button size="lg" variant="secondary" onClick={() => guess(2)} disabled={busy || !ready}>
              {kind === 'rai' ? 'AI' : 'B is hotter'} <Kbd className="max-sm:hidden">2</Kbd>
            </Button>
          </div>
          <div className="mt-3 flex items-center justify-between text-xs text-ink-muted">
            <span>{ready ? 'Your call.' : kind === 'rai' ? 'Have a listen first.' : 'Have a listen to both first.'}</span>
            <button onClick={skip} disabled={busy} className="hover:text-ink">Skip</button>
          </div>
        </>
      ) : (
        <div className="mt-5 border-t border-hairline pt-4">
          {rai && (
            <>
              <p className={cx('text-base font-semibold', rai.correct ? 'text-ok' : 'text-danger')}>
                {rai.correct ? 'You got it.' : 'Fooled you.'} It was {rai.was_ai ? 'AI' : 'a real recording'}.
              </p>
              <p className="mt-1 text-sm text-ink-muted">
                {rai.fooled_pct !== null ? `It fooled ${rai.fooled_pct}% of listeners. ` : ''}
                Your score: <span className="font-medium tabular-nums text-ink">{rai.my_correct} / {rai.my_total}</span>
              </p>
            </>
          )}
          {hot && (
            <p className="text-sm text-ink-muted">
              {hot.agree_pct !== null
                ? <>You agree with <span className="font-medium tabular-nums text-ink">{hot.agree_pct}%</span> of listeners.</>
                : 'Noted. You’re one of the first to judge this pair.'}
            </p>
          )}
          <Button className="mt-4" variant="primary" onClick={onDone}>
            Back to labelling <Kbd className="border-white/30 bg-transparent text-white/80 max-sm:hidden">↵</Kbd>
          </Button>
          {rai && (
            <p className="mt-4 text-[11px] leading-relaxed text-ink-muted">
              Real voices are from the Expresso dataset (Meta AI, CC BY-NC 4.0), recorded for speech research. The AI
              clips are our model copying those same speakers.
            </p>
          )}
        </div>
      )}
    </Card>
  )
}
